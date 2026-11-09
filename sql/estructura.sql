CREATE SCHEMA meteo;

CREATE SCHEMA telemetry;

CREATE FUNCTION telemetry.detectar_gaps(p_desde timestamp with time zone, p_hasta timestamp with time zone, p_umbral interval DEFAULT '00:15:00'::interval, p_registrar boolean DEFAULT false) RETURNS TABLE(deviceuuid uuid, ultimopresente timestamp with time zone, siguientepresente timestamp with time zone, faltantes integer, gapid bigint)
    LANGUAGE plpgsql
    AS $$
#variable_conflict use_column
DECLARE
    r record;
BEGIN
    FOR r IN
        WITH lecturas AS (
            SELECT h.deviceuuid, h.datatime FROM telemetry.inverter_history h
            WHERE h.datatime BETWEEN p_desde AND p_hasta
            UNION ALL
            SELECT m.deviceuuid, m.datatime FROM telemetry.meter_history m
            WHERE m.datatime BETWEEN p_desde AND p_hasta
        ), saltos AS (
            SELECT l.deviceuuid,
                   lag(l.datatime) OVER (PARTITION BY l.deviceuuid ORDER BY l.datatime) AS anterior,
                   l.datatime AS siguiente
            FROM lecturas l
        )
        SELECT s.deviceuuid, s.anterior, s.siguiente
        FROM saltos s
        WHERE s.anterior IS NOT NULL AND s.siguiente - s.anterior > p_umbral
        ORDER BY s.deviceuuid, s.anterior
    LOOP
        deviceuuid        := r.deviceuuid;
        ultimopresente    := r.anterior;
        siguientepresente := r.siguiente;
        faltantes         := greatest(round(extract(epoch FROM r.siguiente - r.anterior) / 300)::integer - 1, 1);
        gapid             := NULL;
        IF p_registrar THEN
            gapid := telemetry.registrar_gap_ventana(
                r.deviceuuid, r.anterior + interval '5 minutes', r.siguiente - interval '5 minutes',
                'deteccion', format('salto de %s entre lecturas', r.siguiente - r.anterior));
        END IF;
        RETURN NEXT;
    END LOOP;
END;
$$;

CREATE FUNCTION telemetry.plantlocaltime_a_utc(p_plantlocaltime timestamp without time zone, p_consulta timestamp with time zone) RETURNS timestamp with time zone
    LANGUAGE plpgsql IMMUTABLE STRICT
    AS $$
DECLARE
    v_desfase_min integer;
    v_utc         timestamptz;
BEGIN
    v_desfase_min := round(extract(epoch FROM p_plantlocaltime - (p_consulta AT TIME ZONE 'UTC')) / 900.0) * 15;
    v_utc := (p_plantlocaltime - make_interval(mins => v_desfase_min)) AT TIME ZONE 'UTC';
    IF abs(extract(epoch FROM p_consulta - v_utc)) > 300 THEN
        RETURN NULL;
    END IF;
    RETURN v_utc;
END;
$$;

CREATE FUNCTION telemetry.registrar_gap(p_deviceuuid uuid, p_datatime timestamp with time zone, p_nota text DEFAULT NULL::text) RETURNS bigint
    LANGUAGE sql
    AS $$
    SELECT telemetry.registrar_gap_ventana(p_deviceuuid, p_datatime, p_datatime, 'manual', p_nota);
$$;

CREATE FUNCTION telemetry.registrar_gap_ventana(p_deviceuuid uuid, p_primero timestamp with time zone, p_ultimo timestamp with time zone, p_origen text DEFAULT 'manual'::text, p_nota text DEFAULT NULL::text) RETURNS bigint
    LANGUAGE plpgsql
    AS $$
DECLARE
    v_inicio timestamptz := p_primero - interval '10 minutes' - interval '150 seconds';
    v_fin    timestamptz := p_ultimo  + interval '10 minutes' + interval '150 seconds';
    v_id     bigint;
BEGIN
    IF p_ultimo < p_primero THEN
        RAISE EXCEPTION 'p_ultimo (%) es anterior a p_primero (%)', p_ultimo, p_primero;
    END IF;
    INSERT INTO telemetry.etl_gaps (deviceuuid, inicio, fin, origen, nota)
    VALUES (p_deviceuuid, v_inicio, v_fin, p_origen, p_nota)
    ON CONFLICT (deviceuuid, inicio, fin) DO NOTHING
    RETURNING gapid INTO v_id;
    IF v_id IS NULL THEN
        SELECT gapid INTO v_id FROM telemetry.etl_gaps
        WHERE deviceuuid = p_deviceuuid AND inicio = v_inicio AND fin = v_fin;
    END IF;
    RETURN v_id;
END;
$$;

CREATE TABLE meteo.om_satellite (
    grupo text NOT NULL,
    modelo text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    shortwave_radiation double precision,
    direct_radiation double precision,
    diffuse_radiation double precision,
    direct_normal_irradiance double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.om_previous_runs (
    grupo text NOT NULL,
    modelo text NOT NULL,
    previousday smallint NOT NULL,
    datatime timestamp with time zone NOT NULL,
    shortwave_radiation double precision,
    direct_normal_irradiance double precision,
    diffuse_radiation double precision,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    cloud_cover double precision,
    precipitation double precision,
    wind_speed_10m double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.om_single_runs (
    grupo text NOT NULL,
    modelo text NOT NULL,
    run timestamp with time zone NOT NULL,
    datatime timestamp with time zone NOT NULL,
    shortwave_radiation double precision,
    direct_normal_irradiance double precision,
    diffuse_radiation double precision,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    cloud_cover double precision,
    cloud_cover_low double precision,
    precipitation double precision,
    wind_speed_10m double precision,
    cape double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.om_historical_forecast (
    grupo text NOT NULL,
    modelo text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    shortwave_radiation double precision,
    direct_radiation double precision,
    diffuse_radiation double precision,
    direct_normal_irradiance double precision,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    dew_point_2m double precision,
    precipitation double precision,
    cloud_cover double precision,
    cloud_cover_low double precision,
    cloud_cover_mid double precision,
    cloud_cover_high double precision,
    surface_pressure double precision,
    wind_speed_10m double precision,
    cape double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.om_air_quality (
    grupo text NOT NULL,
    modelo text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    aerosol_optical_depth double precision,
    dust double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.nasa_power (
    grupo text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    allsky_sfc_sw_dwn double precision,
    clrsky_sfc_sw_dwn double precision,
    allsky_sfc_sw_dni double precision,
    allsky_sfc_sw_diff double precision,
    allsky_kt double precision,
    t2m double precision,
    t2mdew double precision,
    rh2m double precision,
    ws10m double precision,
    wd10m double precision,
    ps double precision,
    prectotcorr double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.nsrdb_goes (
    grupo text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    ghi double precision,
    dni double precision,
    dhi double precision,
    clearsky_ghi double precision,
    clearsky_dni double precision,
    clearsky_dhi double precision,
    cloud_type double precision,
    air_temperature double precision,
    relative_humidity double precision,
    surface_pressure double precision,
    wind_speed double precision,
    total_precipitable_water double precision,
    aod double precision,
    solar_zenith_angle double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE telemetry.plant_realtime (
    datatime timestamp with time zone NOT NULL,
    plantuuid uuid NOT NULL,
    plantlocaltime timestamp without time zone NOT NULL,
    dailyyield double precision,
    totalyield double precision,
    dailycharged double precision,
    totalcharged double precision,
    dailydischarged double precision,
    totaldischarged double precision,
    dailyimported double precision,
    totalimported double precision,
    dailyexported double precision,
    totalexported double precision,
    dailyearnings double precision,
    totalearnings double precision,
    cargatime timestamp with time zone DEFAULT now()
);

CREATE TABLE telemetry.inverter_realtime (
    datatime timestamp with time zone NOT NULL,
    deviceuuid uuid NOT NULL,
    plantlocaltime timestamp without time zone,
    devicestatus smallint,
    acvoltage1 double precision,
    acvoltage2 double precision,
    acvoltage3 double precision,
    accurrent1 double precision,
    accurrent2 double precision,
    accurrent3 double precision,
    acfrequency1 double precision,
    acfrequency2 double precision,
    acfrequency3 double precision,
    acpower1 double precision,
    acpower2 double precision,
    acpower3 double precision,
    gridfrequency double precision,
    totalactivepower double precision,
    totalreactivepower double precision,
    totalpowerfactor double precision,
    invertertemperature double precision,
    dailyyield double precision,
    totalyield double precision,
    dailyacoutput double precision,
    totalacoutput double precision,
    dailyacinput double precision,
    totalacinput double precision,
    mppttotalinputpower double precision,
    mppt1voltage double precision,
    mppt1current double precision,
    mppt1power double precision,
    mppt2voltage double precision,
    mppt2current double precision,
    mppt2power double precision,
    mppt3voltage double precision,
    mppt3current double precision,
    mppt3power double precision,
    mppt4voltage double precision,
    mppt4current double precision,
    mppt4power double precision,
    mppt5voltage double precision,
    mppt5current double precision,
    mppt5power double precision,
    mppt6voltage double precision,
    mppt6current double precision,
    mppt6power double precision,
    mppt7voltage double precision,
    mppt7current double precision,
    mppt7power double precision,
    mppt8voltage double precision,
    mppt8current double precision,
    mppt8power double precision,
    mppt9voltage double precision,
    mppt9current double precision,
    mppt9power double precision,
    mppt10voltage double precision,
    mppt10current double precision,
    mppt10power double precision,
    mppt11voltage double precision,
    mppt11current double precision,
    mppt11power double precision,
    mppt12voltage double precision,
    mppt12current double precision,
    mppt12power double precision,
    pv1voltage double precision,
    pv1current double precision,
    pv1power double precision,
    pv2voltage double precision,
    pv2current double precision,
    pv2power double precision,
    pv3voltage double precision,
    pv3current double precision,
    pv3power double precision,
    pv4voltage double precision,
    pv4current double precision,
    pv4power double precision,
    pv5voltage double precision,
    pv5current double precision,
    pv5power double precision,
    pv6voltage double precision,
    pv6current double precision,
    pv6power double precision,
    pv7voltage double precision,
    pv7current double precision,
    pv7power double precision,
    pv8voltage double precision,
    pv8current double precision,
    pv8power double precision,
    pv9voltage double precision,
    pv9current double precision,
    pv9power double precision,
    pv10voltage double precision,
    pv10current double precision,
    pv10power double precision,
    pv11voltage double precision,
    pv11current double precision,
    pv11power double precision,
    pv12voltage double precision,
    pv12current double precision,
    pv12power double precision,
    pv13voltage double precision,
    pv13current double precision,
    pv13power double precision,
    pv14voltage double precision,
    pv14current double precision,
    pv14power double precision,
    pv15voltage double precision,
    pv15current double precision,
    pv15power double precision,
    pv16voltage double precision,
    pv16current double precision,
    pv16power double precision,
    pv17voltage double precision,
    pv17current double precision,
    pv17power double precision,
    pv18voltage double precision,
    pv18current double precision,
    pv18power double precision,
    pv19voltage double precision,
    pv19current double precision,
    pv19power double precision,
    pv20voltage double precision,
    pv20current double precision,
    pv20power double precision,
    pv21voltage double precision,
    pv21current double precision,
    pv21power double precision,
    pv22voltage double precision,
    pv22current double precision,
    pv22power double precision,
    pv23voltage double precision,
    pv23current double precision,
    pv23power double precision,
    pv24voltage double precision,
    pv24current double precision,
    pv24power double precision,
    gridpower double precision,
    todayimportenergy double precision,
    totalimportenergy double precision,
    todayexportenergy double precision,
    totalexportenergy double precision,
    gridpowerm2 double precision,
    todayimportenergym2 double precision,
    totalimportenergym2 double precision,
    todayexportenergym2 double precision,
    totalexportenergym2 double precision,
    epsl1voltage double precision,
    epsl1current double precision,
    epsl1activepower double precision,
    epsl1apparentpower double precision,
    epsl2voltage double precision,
    epsl2current double precision,
    epsl2activepower double precision,
    epsl2apparentpower double precision,
    epsl3voltage double precision,
    epsl3current double precision,
    epsl3activepower double precision,
    epsl3apparentpower double precision,
    l1l2voltage double precision,
    l2l3voltage double precision,
    l1l3voltage double precision,
    cargatime timestamp with time zone DEFAULT now()
);

CREATE TABLE telemetry.inverter_history (
    datatime timestamp with time zone NOT NULL,
    deviceuuid uuid NOT NULL,
    plantlocaltime timestamp without time zone,
    devicestatus smallint,
    acvoltage1 double precision,
    acvoltage2 double precision,
    acvoltage3 double precision,
    accurrent1 double precision,
    accurrent2 double precision,
    accurrent3 double precision,
    acfrequency1 double precision,
    acfrequency2 double precision,
    acfrequency3 double precision,
    acpower1 double precision,
    acpower2 double precision,
    acpower3 double precision,
    gridfrequency double precision,
    totalactivepower double precision,
    totalreactivepower double precision,
    totalpowerfactor double precision,
    invertertemperature double precision,
    dailyyield double precision,
    totalyield double precision,
    dailyacoutput double precision,
    totalacoutput double precision,
    dailyacinput double precision,
    totalacinput double precision,
    mppttotalinputpower double precision,
    mppt1voltage double precision,
    mppt1current double precision,
    mppt1power double precision,
    mppt2voltage double precision,
    mppt2current double precision,
    mppt2power double precision,
    mppt3voltage double precision,
    mppt3current double precision,
    mppt3power double precision,
    mppt4voltage double precision,
    mppt4current double precision,
    mppt4power double precision,
    mppt5voltage double precision,
    mppt5current double precision,
    mppt5power double precision,
    mppt6voltage double precision,
    mppt6current double precision,
    mppt6power double precision,
    mppt7voltage double precision,
    mppt7current double precision,
    mppt7power double precision,
    mppt8voltage double precision,
    mppt8current double precision,
    mppt8power double precision,
    mppt9voltage double precision,
    mppt9current double precision,
    mppt9power double precision,
    mppt10voltage double precision,
    mppt10current double precision,
    mppt10power double precision,
    mppt11voltage double precision,
    mppt11current double precision,
    mppt11power double precision,
    mppt12voltage double precision,
    mppt12current double precision,
    mppt12power double precision,
    pv1voltage double precision,
    pv1current double precision,
    pv1power double precision,
    pv2voltage double precision,
    pv2current double precision,
    pv2power double precision,
    pv3voltage double precision,
    pv3current double precision,
    pv3power double precision,
    pv4voltage double precision,
    pv4current double precision,
    pv4power double precision,
    pv5voltage double precision,
    pv5current double precision,
    pv5power double precision,
    pv6voltage double precision,
    pv6current double precision,
    pv6power double precision,
    pv7voltage double precision,
    pv7current double precision,
    pv7power double precision,
    pv8voltage double precision,
    pv8current double precision,
    pv8power double precision,
    pv9voltage double precision,
    pv9current double precision,
    pv9power double precision,
    pv10voltage double precision,
    pv10current double precision,
    pv10power double precision,
    pv11voltage double precision,
    pv11current double precision,
    pv11power double precision,
    pv12voltage double precision,
    pv12current double precision,
    pv12power double precision,
    pv13voltage double precision,
    pv13current double precision,
    pv13power double precision,
    pv14voltage double precision,
    pv14current double precision,
    pv14power double precision,
    pv15voltage double precision,
    pv15current double precision,
    pv15power double precision,
    pv16voltage double precision,
    pv16current double precision,
    pv16power double precision,
    pv17voltage double precision,
    pv17current double precision,
    pv17power double precision,
    pv18voltage double precision,
    pv18current double precision,
    pv18power double precision,
    pv19voltage double precision,
    pv19current double precision,
    pv19power double precision,
    pv20voltage double precision,
    pv20current double precision,
    pv20power double precision,
    pv21voltage double precision,
    pv21current double precision,
    pv21power double precision,
    pv22voltage double precision,
    pv22current double precision,
    pv22power double precision,
    pv23voltage double precision,
    pv23current double precision,
    pv23power double precision,
    pv24voltage double precision,
    pv24current double precision,
    pv24power double precision,
    gridpower double precision,
    todayimportenergy double precision,
    totalimportenergy double precision,
    todayexportenergy double precision,
    totalexportenergy double precision,
    gridpowerm2 double precision,
    todayimportenergym2 double precision,
    totalimportenergym2 double precision,
    todayexportenergym2 double precision,
    totalexportenergym2 double precision,
    epsl1voltage double precision,
    epsl1current double precision,
    epsl1activepower double precision,
    epsl1apparentpower double precision,
    epsl2voltage double precision,
    epsl2current double precision,
    epsl2activepower double precision,
    epsl2apparentpower double precision,
    epsl3voltage double precision,
    epsl3current double precision,
    epsl3activepower double precision,
    epsl3apparentpower double precision,
    l1l2voltage double precision,
    l2l3voltage double precision,
    l1l3voltage double precision,
    cargatime timestamp with time zone DEFAULT now()
);

CREATE TABLE telemetry.meter_realtime (
    datatime timestamp with time zone NOT NULL,
    deviceuuid uuid NOT NULL,
    plantlocaltime timestamp without time zone,
    importenergy double precision,
    exportenergy double precision,
    totalactivepower double precision,
    totalreactivepower double precision,
    totalapparentpower double precision,
    powerfactor double precision,
    gridfrequency double precision,
    acfrequency1 double precision,
    acfrequency2 double precision,
    acfrequency3 double precision,
    l1voltage double precision,
    l1current double precision,
    l1activepower double precision,
    l1reactivepower double precision,
    l2voltage double precision,
    l2current double precision,
    l2activepower double precision,
    l2reactivepower double precision,
    l3voltage double precision,
    l3current double precision,
    l3activepower double precision,
    l3reactivepower double precision,
    cargatime timestamp with time zone DEFAULT now()
);

CREATE TABLE telemetry.meter_history (
    datatime timestamp with time zone NOT NULL,
    deviceuuid uuid NOT NULL,
    plantlocaltime timestamp without time zone,
    importenergy double precision,
    exportenergy double precision,
    totalactivepower double precision,
    totalreactivepower double precision,
    totalapparentpower double precision,
    powerfactor double precision,
    gridfrequency double precision,
    acfrequency1 double precision,
    acfrequency2 double precision,
    acfrequency3 double precision,
    l1voltage double precision,
    l1current double precision,
    l1activepower double precision,
    l1reactivepower double precision,
    l2voltage double precision,
    l2current double precision,
    l2activepower double precision,
    l2reactivepower double precision,
    l3voltage double precision,
    l3current double precision,
    l3activepower double precision,
    l3reactivepower double precision,
    cargatime timestamp with time zone DEFAULT now()
);

CREATE TABLE telemetry.etl_ciclos (
    servicio text NOT NULL,
    iniciotime timestamp with time zone NOT NULL,
    fintime timestamp with time zone,
    estado text NOT NULL,
    duracions double precision,
    elementos integer,
    insertadas integer,
    duplicadas integer,
    rechazadas integer,
    errores integer,
    tokenorigen text,
    sincronizacion boolean,
    etapafallida text,
    errortipo text,
    errorcodigo integer,
    errorresumen text,
    errormensaje text,
    tautenticacion double precision,
    tconexionbd double precision,
    tsincronizacion double precision,
    tconsultas double precision,
    tconsultamedia double precision,
    tconsultamax double precision,
    llamadasapi integer,
    sondeos integer,
    sondeoscondatos integer,
    gapsresueltos integer,
    gapsagotados integer,
    CONSTRAINT ck_etl_ciclos_estado CHECK ((estado = ANY (ARRAY[('OKOK'::character varying)::text, ('WARN'::character varying)::text, ('FAIL'::character varying)::text])))
);

CREATE TABLE telemetry.etl_incidencias (
    servicio text NOT NULL,
    iniciotime timestamp with time zone NOT NULL,
    orden smallint NOT NULL,
    eventotime timestamp with time zone NOT NULL,
    estado text NOT NULL,
    etiqueta text NOT NULL,
    etapa text,
    motivo text,
    plantuuid uuid,
    deviceuuid uuid,
    idhuella text,
    errortipo text,
    errorcodigo integer,
    errorresumen text,
    errormensaje text,
    duracions double precision,
    detalle text,
    CONSTRAINT ck_etl_incidencias_estado CHECK ((estado = ANY (ARRAY[('WARN'::character varying)::text, ('FAIL'::character varying)::text])))
);

CREATE TABLE meteo.ideam_observaciones (
    codigoestacion text NOT NULL,
    codigosensor text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    fechaobservacion timestamp without time zone NOT NULL,
    valorobservado double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.om_archive (
    grupo text NOT NULL,
    modelo text NOT NULL,
    datatime timestamp with time zone NOT NULL,
    shortwave_radiation double precision,
    direct_radiation double precision,
    diffuse_radiation double precision,
    direct_normal_irradiance double precision,
    terrestrial_radiation double precision,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    dew_point_2m double precision,
    precipitation double precision,
    cloud_cover double precision,
    cloud_cover_low double precision,
    cloud_cover_mid double precision,
    cloud_cover_high double precision,
    surface_pressure double precision,
    wind_speed_10m double precision,
    wind_direction_10m double precision,
    wind_gusts_10m double precision,
    sunshine_duration double precision,
    vapour_pressure_deficit double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.archivos_importados (
    sha256 text NOT NULL,
    archivo text NOT NULL,
    llamador smallint,
    fuente text NOT NULL,
    modelo text NOT NULL,
    grupo text NOT NULL,
    inicio date,
    fin date,
    run timestamp with time zone,
    filas integer NOT NULL,
    importado timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.bloques_no_disponibles (
    fuente text NOT NULL,
    modelo text NOT NULL,
    grupo text NOT NULL,
    bloque text NOT NULL,
    run timestamp with time zone,
    motivo text,
    detectado timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.celdas (
    grupo text NOT NULL,
    fuente text NOT NULL,
    modelo text NOT NULL,
    latitude double precision,
    longitude double precision,
    elevation double precision,
    primeracarga timestamp with time zone DEFAULT now() NOT NULL,
    ultimacarga timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.enso_oni (
    anio smallint NOT NULL,
    mes smallint NOT NULL,
    temporada text NOT NULL,
    sst_total numeric(6,2),
    oni numeric(5,2) NOT NULL,
    cargatime timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT enso_oni_mes_check CHECK (((mes >= 1) AND (mes <= 12)))
);

CREATE TABLE meteo.fuentes (
    fuente text NOT NULL,
    proveedor text NOT NULL,
    endpoint text NOT NULL,
    tipo text NOT NULL,
    resolucionespacial text,
    resoluciontemporal text,
    latencia text,
    autenticacion text,
    licencia text,
    tabladestino text NOT NULL,
    descripcion text,
    CONSTRAINT ck_fuentes_tipo CHECK ((tipo = ANY (ARRAY['reanalisis'::text, 'satelite'::text, 'pronostico_archivado'::text, 'pronostico'::text, 'aerosoles'::text, 'estacion'::text])))
);

CREATE TABLE meteo.grupos (
    grupo text NOT NULL,
    zona text NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    diametrokm double precision NOT NULL,
    prioridad smallint NOT NULL,
    llamador smallint NOT NULL,
    nplantas smallint NOT NULL,
    ninversores smallint NOT NULL,
    criterio text,
    creado timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.ideam_estaciones (
    codigoestacion text NOT NULL,
    nombreestacion text,
    categoria text,
    tecnologia text,
    estado text,
    departamento text,
    municipio text,
    latitud double precision,
    longitud double precision,
    altitud double precision,
    cargatime timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE meteo.ideam_grupos_estaciones (
    grupo text NOT NULL,
    codigoestacion text NOT NULL,
    distanciakm double precision NOT NULL
);

CREATE TABLE meteo.ideam_sensores (
    codigosensor text NOT NULL,
    descripcionsensor text,
    unidadmedida text,
    dataset text
);

CREATE TABLE meteo.modelos (
    fuente text NOT NULL,
    modelo text NOT NULL,
    descripcion text,
    resolucion text,
    disponibledesde date,
    nota text
);

CREATE TABLE meteo.plantas_grupos (
    plantuuid uuid NOT NULL,
    grupo text NOT NULL,
    distanciakm double precision NOT NULL
);

CREATE VIEW meteo.v_cobertura WITH (security_invoker='true') AS
 SELECT 'om_archive'::text AS fuente,
    om_archive.modelo,
    om_archive.grupo,
    min(om_archive.datatime) AS primerdato,
    max(om_archive.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(om_archive.datatime))) / 3600.0), 1) AS atrasohoras,
    max(om_archive.cargatime) AS ultimacarga
   FROM meteo.om_archive
  GROUP BY om_archive.modelo, om_archive.grupo
UNION ALL
 SELECT 'om_satellite'::text AS fuente,
    om_satellite.modelo,
    om_satellite.grupo,
    min(om_satellite.datatime) AS primerdato,
    max(om_satellite.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(om_satellite.datatime))) / 3600.0), 1) AS atrasohoras,
    max(om_satellite.cargatime) AS ultimacarga
   FROM meteo.om_satellite
  GROUP BY om_satellite.modelo, om_satellite.grupo
UNION ALL
 SELECT 'om_previous_runs'::text AS fuente,
    om_previous_runs.modelo,
    om_previous_runs.grupo,
    min(om_previous_runs.datatime) AS primerdato,
    max(om_previous_runs.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(om_previous_runs.datatime))) / 3600.0), 1) AS atrasohoras,
    max(om_previous_runs.cargatime) AS ultimacarga
   FROM meteo.om_previous_runs
  GROUP BY om_previous_runs.modelo, om_previous_runs.grupo
UNION ALL
 SELECT 'om_single_runs'::text AS fuente,
    om_single_runs.modelo,
    om_single_runs.grupo,
    min(om_single_runs.datatime) AS primerdato,
    max(om_single_runs.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(om_single_runs.datatime))) / 3600.0), 1) AS atrasohoras,
    max(om_single_runs.cargatime) AS ultimacarga
   FROM meteo.om_single_runs
  GROUP BY om_single_runs.modelo, om_single_runs.grupo
UNION ALL
 SELECT 'om_historical_forecast'::text AS fuente,
    om_historical_forecast.modelo,
    om_historical_forecast.grupo,
    min(om_historical_forecast.datatime) AS primerdato,
    max(om_historical_forecast.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(om_historical_forecast.datatime))) / 3600.0), 1) AS atrasohoras,
    max(om_historical_forecast.cargatime) AS ultimacarga
   FROM meteo.om_historical_forecast
  GROUP BY om_historical_forecast.modelo, om_historical_forecast.grupo
UNION ALL
 SELECT 'om_air_quality'::text AS fuente,
    om_air_quality.modelo,
    om_air_quality.grupo,
    min(om_air_quality.datatime) AS primerdato,
    max(om_air_quality.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(om_air_quality.datatime))) / 3600.0), 1) AS atrasohoras,
    max(om_air_quality.cargatime) AS ultimacarga
   FROM meteo.om_air_quality
  GROUP BY om_air_quality.modelo, om_air_quality.grupo
UNION ALL
 SELECT 'nasa_power'::text AS fuente,
    'nasa_power'::text AS modelo,
    nasa_power.grupo,
    min(nasa_power.datatime) AS primerdato,
    max(nasa_power.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(nasa_power.datatime))) / 3600.0), 1) AS atrasohoras,
    max(nasa_power.cargatime) AS ultimacarga
   FROM meteo.nasa_power
  GROUP BY 'nasa_power'::text, nasa_power.grupo
UNION ALL
 SELECT 'nsrdb_goes'::text AS fuente,
    'nsrdb_goes'::text AS modelo,
    nsrdb_goes.grupo,
    min(nsrdb_goes.datatime) AS primerdato,
    max(nsrdb_goes.datatime) AS ultimodato,
    count(*) AS filas,
    round((EXTRACT(epoch FROM (now() - max(nsrdb_goes.datatime))) / 3600.0), 1) AS atrasohoras,
    max(nsrdb_goes.cargatime) AS ultimacarga
   FROM meteo.nsrdb_goes
  GROUP BY 'nsrdb_goes'::text, nsrdb_goes.grupo;

CREATE VIEW meteo.v_huecos WITH (security_invoker='true') AS
 WITH saltos AS (
         SELECT 'om_archive'::text AS fuente,
            om_archive.modelo,
            om_archive.grupo,
            NULL::smallint AS previousday,
            om_archive.datatime,
            lead(om_archive.datatime) OVER (PARTITION BY om_archive.grupo, om_archive.modelo ORDER BY om_archive.datatime) AS siguiente,
            '01:00:00'::interval AS paso
           FROM meteo.om_archive
        UNION ALL
         SELECT 'om_satellite'::text,
            om_satellite.modelo,
            om_satellite.grupo,
            NULL::smallint,
            om_satellite.datatime,
            lead(om_satellite.datatime) OVER (PARTITION BY om_satellite.grupo, om_satellite.modelo ORDER BY om_satellite.datatime) AS lead,
            '00:15:00'::interval AS "interval"
           FROM meteo.om_satellite
        UNION ALL
         SELECT 'om_previous_runs'::text,
            om_previous_runs.modelo,
            om_previous_runs.grupo,
            om_previous_runs.previousday,
            om_previous_runs.datatime,
            lead(om_previous_runs.datatime) OVER (PARTITION BY om_previous_runs.grupo, om_previous_runs.modelo, om_previous_runs.previousday ORDER BY om_previous_runs.datatime) AS lead,
            '01:00:00'::interval AS "interval"
           FROM meteo.om_previous_runs
        UNION ALL
         SELECT 'om_historical_forecast'::text,
            om_historical_forecast.modelo,
            om_historical_forecast.grupo,
            NULL::smallint,
            om_historical_forecast.datatime,
            lead(om_historical_forecast.datatime) OVER (PARTITION BY om_historical_forecast.grupo, om_historical_forecast.modelo ORDER BY om_historical_forecast.datatime) AS lead,
            '01:00:00'::interval AS "interval"
           FROM meteo.om_historical_forecast
        UNION ALL
         SELECT 'om_air_quality'::text,
            om_air_quality.modelo,
            om_air_quality.grupo,
            NULL::smallint,
            om_air_quality.datatime,
            lead(om_air_quality.datatime) OVER (PARTITION BY om_air_quality.grupo, om_air_quality.modelo ORDER BY om_air_quality.datatime) AS lead,
            '01:00:00'::interval AS "interval"
           FROM meteo.om_air_quality
        UNION ALL
         SELECT 'nasa_power'::text,
            'power_hourly'::text,
            nasa_power.grupo,
            NULL::smallint,
            nasa_power.datatime,
            lead(nasa_power.datatime) OVER (PARTITION BY nasa_power.grupo ORDER BY nasa_power.datatime) AS lead,
            '01:00:00'::interval AS "interval"
           FROM meteo.nasa_power
        UNION ALL
         SELECT 'nsrdb_goes'::text,
            'goes_v4'::text,
            nsrdb_goes.grupo,
            NULL::smallint,
            nsrdb_goes.datatime,
            lead(nsrdb_goes.datatime) OVER (PARTITION BY nsrdb_goes.grupo ORDER BY nsrdb_goes.datatime) AS lead,
            '00:10:00'::interval AS "interval"
           FROM meteo.nsrdb_goes
        ), corridas AS (
         SELECT r.modelo,
            r.grupo,
            r.run,
            lead(r.run) OVER (PARTITION BY r.grupo, r.modelo ORDER BY r.run) AS siguiente
           FROM ( SELECT om_single_runs.modelo,
                    om_single_runs.grupo,
                    om_single_runs.run
                   FROM meteo.om_single_runs
                  GROUP BY om_single_runs.modelo, om_single_runs.grupo, om_single_runs.run) r
        )
 SELECT saltos.fuente,
    saltos.modelo,
    saltos.grupo,
    saltos.previousday,
    (saltos.datatime + saltos.paso) AS desde,
    (saltos.siguiente - saltos.paso) AS hasta,
    ((saltos.siguiente - saltos.datatime) - saltos.paso) AS duracion,
    (((EXTRACT(epoch FROM (saltos.siguiente - saltos.datatime)) / EXTRACT(epoch FROM saltos.paso)))::integer - 1) AS pasosfaltantes,
    'interno'::text AS tipo
   FROM saltos
  WHERE ((saltos.siguiente - saltos.datatime) > saltos.paso)
UNION ALL
 SELECT 'om_single_runs'::text AS fuente,
    c.modelo,
    c.grupo,
    NULL::smallint AS previousday,
    (c.run + '12:00:00'::interval) AS desde,
    (c.siguiente - '12:00:00'::interval) AS hasta,
    ((c.siguiente - c.run) - '12:00:00'::interval) AS duracion,
    (((EXTRACT(epoch FROM (c.siguiente - c.run)) / (43200)::numeric))::integer - 1) AS pasosfaltantes,
        CASE
            WHEN (EXISTS ( SELECT 1
               FROM meteo.bloques_no_disponibles n
              WHERE ((n.fuente = 'om_single_runs'::text) AND (n.grupo = c.grupo) AND (n.modelo = c.modelo) AND ((n.run >= (c.run + '12:00:00'::interval)) AND (n.run <= (c.siguiente - '12:00:00'::interval)))))) THEN 'no_disponible_en_fuente'::text
            ELSE 'corrida_faltante'::text
        END AS tipo
   FROM corridas c
  WHERE ((c.siguiente - c.run) > '12:00:00'::interval);

CREATE TABLE meteo.variables (
    fuente text NOT NULL,
    variable text NOT NULL,
    unidad text NOT NULL,
    tipovalor text NOT NULL,
    marcatiempo text NOT NULL,
    periodominutos smallint NOT NULL,
    equivalente text,
    descripcion text,
    CONSTRAINT ck_variables_marcatiempo CHECK ((marcatiempo = ANY (ARRAY['inicio'::text, 'fin'::text, 'instante'::text]))),
    CONSTRAINT ck_variables_tipovalor CHECK ((tipovalor = ANY (ARRAY['media'::text, 'instantaneo'::text, 'acumulado'::text, 'maximo'::text])))
);

CREATE TABLE telemetry.devices (
    deviceuuid uuid DEFAULT gen_random_uuid() NOT NULL,
    plantuuid uuid NOT NULL,
    devicetype smallint NOT NULL,
    devicemodel smallint,
    armversion character varying(16),
    dspversion character varying(16),
    ratedpower double precision,
    onlinestatus smallint,
    flag smallint
);

CREATE TABLE telemetry.plants (
    plantuuid uuid DEFAULT gen_random_uuid() NOT NULL,
    cityid character varying(10) DEFAULT 'CO_47001'::character varying NOT NULL,
    businesstype smallint NOT NULL,
    batterycapacity double precision,
    pvcapacity double precision,
    createtime timestamp with time zone,
    planttimezone character varying(100),
    plantstate smallint,
    electricitypriceunit character varying(3),
    latitude double precision,
    longitude double precision,
    altitude numeric(6,2) DEFAULT NULL::numeric,
    CONSTRAINT ck_plants_latitude_redondeada CHECK ((latitude = (round((latitude)::numeric, 2))::double precision)),
    CONSTRAINT ck_plants_longitude_redondeada CHECK ((longitude = (round((longitude)::numeric, 2))::double precision))
);

CREATE TABLE telemetry.business_types (
    businesstype smallint NOT NULL,
    description character varying(60) NOT NULL
);

CREATE TABLE telemetry.cities (
    cityid character varying(10) NOT NULL,
    cityname character varying(100) NOT NULL,
    country character varying(60) NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL
);

CREATE TABLE telemetry.device_types (
    devicetype smallint NOT NULL,
    description character varying(60) NOT NULL
);

CREATE TABLE telemetry.devices_fv (
    deviceuuid uuid NOT NULL,
    kwpdc double precision,
    inclinacion double precision,
    azimut double precision,
    origen text DEFAULT 'estimado'::text NOT NULL,
    excluir boolean DEFAULT false NOT NULL,
    motivoexclusion text,
    actualizado timestamp with time zone DEFAULT now() NOT NULL,
    nota text,
    CONSTRAINT ck_devices_fv_azimut CHECK (((azimut IS NULL) OR ((azimut >= (0)::double precision) AND (azimut < (360)::double precision)))),
    CONSTRAINT ck_devices_fv_inclinacion CHECK (((inclinacion IS NULL) OR ((inclinacion >= (0)::double precision) AND (inclinacion <= (90)::double precision)))),
    CONSTRAINT ck_devices_fv_origen CHECK ((origen = ANY (ARRAY['estimado'::text, 'instalador'::text, 'manual'::text])))
);

CREATE TABLE telemetry.etl_gaps (
    gapid bigint NOT NULL,
    deviceuuid uuid NOT NULL,
    inicio timestamp with time zone NOT NULL,
    fin timestamp with time zone NOT NULL,
    origen text NOT NULL,
    estado text DEFAULT 'pendiente'::text NOT NULL,
    intentos integer DEFAULT 0 NOT NULL,
    ultimointento timestamp with time zone,
    creacion timestamp with time zone DEFAULT now() NOT NULL,
    resolucion timestamp with time zone,
    nota text,
    CONSTRAINT ck_etl_gaps_estado CHECK ((estado = ANY (ARRAY['pendiente'::text, 'resuelto'::text, 'agotado'::text]))),
    CONSTRAINT ck_etl_gaps_origen CHECK ((origen = ANY (ARRAY['manual'::text, 'deteccion'::text]))),
    CONSTRAINT ck_etl_gaps_ventana CHECK ((fin > inicio))
);

ALTER TABLE telemetry.etl_gaps ALTER COLUMN gapid ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME telemetry.etl_gaps_gapid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);

CREATE TABLE telemetry.etl_servicios (
    servicio text NOT NULL,
    rutaapi text NOT NULL,
    tabladestino text NOT NULL,
    elemento text NOT NULL,
    descripcion text,
    CONSTRAINT ck_etl_servicios_elemento CHECK ((elemento = ANY (ARRAY['planta'::text, 'dispositivo'::text, 'ciudad'::text, 'grupo'::text, 'otro'::text])))
);

CREATE TABLE telemetry.modulos_fv (
    modelo text NOT NULL,
    fabricante text NOT NULL,
    pmax_w double precision NOT NULL,
    vmp_v double precision,
    imp_a double precision,
    voc_v double precision,
    isc_a double precision,
    gamma_pmax_pct_c double precision,
    noct_c double precision,
    eficiencia_pct double precision,
    caras text,
    tecnologia text,
    bifacialidad_pct double precision,
    degradacion_primer_anio_pct double precision,
    degradacion_anual_pct double precision,
    datasheet text,
    CONSTRAINT modulos_fv_pmax_w_check CHECK ((pmax_w > (0)::double precision))
);

CREATE TABLE telemetry.plantas_eventos (
    plantuuid uuid NOT NULL,
    tipo text NOT NULL,
    desde date NOT NULL,
    hasta date,
    limite_w double precision,
    detalle text,
    fuente text DEFAULT 'instalador'::text NOT NULL,
    registrado timestamp with time zone DEFAULT now() NOT NULL,
    conformidad_or date,
    vigente boolean,
    CONSTRAINT plantas_eventos_tipo_check CHECK ((tipo = ANY (ARRAY['control_exportacion'::text, 'ampliacion'::text, 'cambio_inversor'::text, 'reparacion'::text, 'otro'::text])))
);

CREATE TABLE telemetry.plantas_fv (
    plantuuid uuid NOT NULL,
    departamento text,
    ciudad text,
    region_natural text,
    altitud_dem_m double precision,
    altitud_ge_m double precision,
    tipo text,
    paneles integer,
    caras_panel text,
    material_celdas text,
    modulo text,
    wp_panel double precision,
    kwp_dc double precision,
    paneles_origen text,
    wp_panel_origen text,
    kwp_dc_origen text,
    tipo_origen text,
    inversor_externo text,
    CONSTRAINT plantas_fv_kwp_dc_check CHECK (((kwp_dc IS NULL) OR (kwp_dc > (0)::double precision))),
    CONSTRAINT plantas_fv_paneles_check CHECK (((paneles IS NULL) OR (paneles > 0)))
);

CREATE TABLE telemetry.plantas_fv_caras (
    plantuuid uuid NOT NULL,
    cara smallint NOT NULL,
    azimut double precision NOT NULL,
    inclinacion double precision NOT NULL,
    paneles integer,
    fraccion double precision NOT NULL,
    techo text,
    CONSTRAINT plantas_fv_caras_azimut_check CHECK (((azimut >= (0)::double precision) AND (azimut < (360)::double precision))),
    CONSTRAINT plantas_fv_caras_fraccion_check CHECK (((fraccion > (0)::double precision) AND (fraccion <= (1)::double precision))),
    CONSTRAINT plantas_fv_caras_inclinacion_check CHECK (((inclinacion >= (0)::double precision) AND (inclinacion <= (90)::double precision)))
);

CREATE TABLE telemetry.plantas_fv_etapas (
    plantuuid uuid NOT NULL,
    desde date NOT NULL,
    paneles integer NOT NULL,
    CONSTRAINT plantas_fv_etapas_paneles_check CHECK ((paneles > 0))
);

CREATE VIEW telemetry.v_estado_dispositivos WITH (security_invoker='true') AS
 SELECT d.deviceuuid,
    d.plantuuid,
    d.devicetype,
    d.onlinestatus,
    h.ultimahistorico,
    round((EXTRACT(epoch FROM (now() - h.ultimahistorico)) / 3600.0), 1) AS atrasohoras,
    r.ultimarealtime,
        CASE
            WHEN (h.ultimahistorico IS NULL) THEN 'sin_datos'::text
            WHEN ((now() - h.ultimahistorico) <= '2 days'::interval) THEN 'historico'::text
            ELSE 'sondeo'::text
        END AS modoesperado,
    (( SELECT count(*) AS count
           FROM telemetry.etl_gaps g
          WHERE ((g.deviceuuid = d.deviceuuid) AND (g.estado = 'pendiente'::text))))::integer AS gapspendientes
   FROM ((telemetry.devices d
     CROSS JOIN LATERAL ( SELECT
                CASE d.devicetype
                    WHEN 3 THEN ( SELECT max(m.datatime) AS max
                       FROM telemetry.meter_history m
                      WHERE (m.deviceuuid = d.deviceuuid))
                    ELSE ( SELECT max(i.datatime) AS max
                       FROM telemetry.inverter_history i
                      WHERE (i.deviceuuid = d.deviceuuid))
                END AS ultimahistorico) h)
     CROSS JOIN LATERAL ( SELECT
                CASE d.devicetype
                    WHEN 3 THEN ( SELECT max(m.datatime) AS max
                       FROM telemetry.meter_realtime m
                      WHERE (m.deviceuuid = d.deviceuuid))
                    ELSE ( SELECT max(i.datatime) AS max
                       FROM telemetry.inverter_realtime i
                      WHERE (i.deviceuuid = d.deviceuuid))
                END AS ultimarealtime) r);

CREATE VIEW telemetry.v_etl_gaps WITH (security_invoker='true') AS
 SELECT g.gapid,
    g.deviceuuid,
    g.inicio,
    g.fin,
    g.origen,
    g.estado,
    g.intentos,
    g.ultimointento,
    g.creacion,
    g.resolucion,
    g.nota,
    d.devicetype,
    (round((EXTRACT(epoch FROM (g.fin - g.inicio)) / (300)::numeric)))::integer AS esperados,
    p.presentes,
    GREATEST(((round((EXTRACT(epoch FROM (g.fin - g.inicio)) / (300)::numeric)))::integer - p.presentes), 0) AS faltantes,
    p.haydatosdespues
   FROM ((telemetry.etl_gaps g
     JOIN telemetry.devices d ON ((d.deviceuuid = g.deviceuuid)))
     CROSS JOIN LATERAL ( SELECT (
                CASE d.devicetype
                    WHEN 3 THEN ( SELECT count(*) AS count
                       FROM telemetry.meter_history m
                      WHERE ((m.deviceuuid = g.deviceuuid) AND ((m.datatime >= g.inicio) AND (m.datatime <= g.fin))))
                    ELSE ( SELECT count(*) AS count
                       FROM telemetry.inverter_history i
                      WHERE ((i.deviceuuid = g.deviceuuid) AND ((i.datatime >= g.inicio) AND (i.datatime <= g.fin))))
                END)::integer AS presentes,
                CASE d.devicetype
                    WHEN 3 THEN (EXISTS ( SELECT 1
                       FROM telemetry.meter_history m
                      WHERE ((m.deviceuuid = g.deviceuuid) AND (m.datatime > g.fin))))
                    ELSE (EXISTS ( SELECT 1
                       FROM telemetry.inverter_history i
                      WHERE ((i.deviceuuid = g.deviceuuid) AND (i.datatime > g.fin))))
                END AS haydatosdespues) p);

CREATE VIEW telemetry.v_nomenclatura WITH (security_invoker='true') AS
 WITH plantas AS (
         SELECT p.plantuuid,
            row_number() OVER (ORDER BY p.createtime, (p.plantuuid)::text) AS numero_planta,
            count(*) OVER () AS total_plantas
           FROM telemetry.plants p
        ), inversores AS (
         SELECT d.deviceuuid,
            d.plantuuid,
            ( SELECT min(h.datatime) AS min
                   FROM telemetry.inverter_history h
                  WHERE (h.deviceuuid = d.deviceuuid)) AS primer_dato
           FROM telemetry.devices d
          WHERE (d.devicetype = 1)
        ), numerados AS (
         SELECT i.deviceuuid,
            i.plantuuid,
            i.primer_dato,
            row_number() OVER (PARTITION BY i.plantuuid ORDER BY i.primer_dato, (i.deviceuuid)::text) AS numero,
            count(*) OVER (PARTITION BY i.plantuuid) AS total
           FROM inversores i
        ), codigos AS (
         SELECT p.plantuuid,
            ('P'::text || lpad((p.numero_planta)::text, GREATEST(2, length((p.total_plantas)::text)), '0'::text)) AS codigo_planta
           FROM plantas p
        )
 SELECT c.plantuuid,
    c.codigo_planta,
    n.deviceuuid,
    n.numero,
    n.total,
    n.primer_dato,
        CASE
            WHEN (n.deviceuuid IS NULL) THEN NULL::text
            ELSE (((c.codigo_planta || 'I'::text) || lpad((n.numero)::text, 2, '0'::text)) || lpad((n.total)::text, 2, '0'::text))
        END AS codigo_inversor
   FROM (codigos c
     LEFT JOIN numerados n USING (plantuuid));

CREATE VIEW telemetry.v_sugerencia_ciudad WITH (security_invoker='true') AS
 SELECT DISTINCT ON (p.plantuuid) p.plantuuid,
    p.cityid AS cityidactual,
    c.cityid AS cityidsugerida,
    c.cityname AS citynamesugerida,
    round(((((2 * 6371))::double precision * asin(sqrt((power(sin((radians((c.latitude - p.latitude)) / (2)::double precision)), (2)::double precision) + ((cos(radians(p.latitude)) * cos(radians(c.latitude))) * power(sin((radians((c.longitude - p.longitude)) / (2)::double precision)), (2)::double precision)))))))::numeric, 1) AS distanciakm,
    ((p.cityid)::text IS DISTINCT FROM (c.cityid)::text) AS difiere
   FROM (telemetry.plants p
     CROSS JOIN telemetry.cities c)
  WHERE ((p.latitude IS NOT NULL) AND (p.longitude IS NOT NULL))
  ORDER BY p.plantuuid, (power(sin((radians((c.latitude - p.latitude)) / (2)::double precision)), (2)::double precision) + ((cos(radians(p.latitude)) * cos(radians(c.latitude))) * power(sin((radians((c.longitude - p.longitude)) / (2)::double precision)), (2)::double precision)));

ALTER TABLE ONLY meteo.archivos_importados
    ADD CONSTRAINT pk_archivos_importados PRIMARY KEY (sha256);

ALTER TABLE ONLY meteo.bloques_no_disponibles
    ADD CONSTRAINT pk_bloques_no_disponibles PRIMARY KEY (fuente, modelo, grupo, bloque);

ALTER TABLE ONLY meteo.celdas
    ADD CONSTRAINT pk_celdas PRIMARY KEY (grupo, fuente, modelo);

ALTER TABLE ONLY meteo.enso_oni
    ADD CONSTRAINT pk_enso_oni PRIMARY KEY (anio, mes);

ALTER TABLE ONLY meteo.fuentes
    ADD CONSTRAINT pk_fuentes PRIMARY KEY (fuente);

ALTER TABLE ONLY meteo.grupos
    ADD CONSTRAINT pk_grupos PRIMARY KEY (grupo);

ALTER TABLE ONLY meteo.ideam_estaciones
    ADD CONSTRAINT pk_ideam_estaciones PRIMARY KEY (codigoestacion);

ALTER TABLE ONLY meteo.ideam_grupos_estaciones
    ADD CONSTRAINT pk_ideam_grupos_estaciones PRIMARY KEY (grupo, codigoestacion);

ALTER TABLE ONLY meteo.ideam_observaciones
    ADD CONSTRAINT pk_ideam_observaciones PRIMARY KEY (codigoestacion, codigosensor, datatime);

ALTER TABLE ONLY meteo.ideam_sensores
    ADD CONSTRAINT pk_ideam_sensores PRIMARY KEY (codigosensor);

ALTER TABLE ONLY meteo.modelos
    ADD CONSTRAINT pk_modelos PRIMARY KEY (fuente, modelo);

ALTER TABLE ONLY meteo.nasa_power
    ADD CONSTRAINT pk_nasa_power PRIMARY KEY (grupo, datatime);

ALTER TABLE ONLY meteo.nsrdb_goes
    ADD CONSTRAINT pk_nsrdb_goes PRIMARY KEY (grupo, datatime);

ALTER TABLE ONLY meteo.om_air_quality
    ADD CONSTRAINT pk_om_air_quality PRIMARY KEY (grupo, modelo, datatime);

ALTER TABLE ONLY meteo.om_archive
    ADD CONSTRAINT pk_om_archive PRIMARY KEY (grupo, modelo, datatime);

ALTER TABLE ONLY meteo.om_historical_forecast
    ADD CONSTRAINT pk_om_historical_forecast PRIMARY KEY (grupo, modelo, datatime);

ALTER TABLE ONLY meteo.om_previous_runs
    ADD CONSTRAINT pk_om_previous_runs PRIMARY KEY (grupo, modelo, previousday, datatime);

ALTER TABLE ONLY meteo.om_satellite
    ADD CONSTRAINT pk_om_satellite PRIMARY KEY (grupo, modelo, datatime);

ALTER TABLE ONLY meteo.om_single_runs
    ADD CONSTRAINT pk_om_single_runs PRIMARY KEY (grupo, modelo, run, datatime);

ALTER TABLE ONLY meteo.plantas_grupos
    ADD CONSTRAINT pk_plantas_grupos PRIMARY KEY (plantuuid);

ALTER TABLE ONLY meteo.variables
    ADD CONSTRAINT pk_variables PRIMARY KEY (fuente, variable);

ALTER TABLE ONLY telemetry.modulos_fv
    ADD CONSTRAINT modulos_fv_pkey PRIMARY KEY (modelo);

ALTER TABLE ONLY telemetry.business_types
    ADD CONSTRAINT pk_business_types PRIMARY KEY (businesstype);

ALTER TABLE ONLY telemetry.cities
    ADD CONSTRAINT pk_cities PRIMARY KEY (cityid);

ALTER TABLE ONLY telemetry.device_types
    ADD CONSTRAINT pk_device_types PRIMARY KEY (devicetype);

ALTER TABLE ONLY telemetry.devices
    ADD CONSTRAINT pk_devices PRIMARY KEY (deviceuuid);

ALTER TABLE ONLY telemetry.devices_fv
    ADD CONSTRAINT pk_devices_fv PRIMARY KEY (deviceuuid);

ALTER TABLE ONLY telemetry.etl_ciclos
    ADD CONSTRAINT pk_etl_ciclos PRIMARY KEY (servicio, iniciotime);

ALTER TABLE ONLY telemetry.etl_gaps
    ADD CONSTRAINT pk_etl_gaps PRIMARY KEY (gapid);

ALTER TABLE ONLY telemetry.etl_incidencias
    ADD CONSTRAINT pk_etl_incidencias PRIMARY KEY (servicio, iniciotime, orden);

ALTER TABLE ONLY telemetry.etl_servicios
    ADD CONSTRAINT pk_etl_servicios PRIMARY KEY (servicio);

ALTER TABLE ONLY telemetry.inverter_history
    ADD CONSTRAINT pk_inverter_history PRIMARY KEY (deviceuuid, datatime);

ALTER TABLE ONLY telemetry.inverter_realtime
    ADD CONSTRAINT pk_inverter_realtime PRIMARY KEY (deviceuuid, datatime);

ALTER TABLE ONLY telemetry.meter_history
    ADD CONSTRAINT pk_meter_history PRIMARY KEY (deviceuuid, datatime);

ALTER TABLE ONLY telemetry.meter_realtime
    ADD CONSTRAINT pk_meter_realtime PRIMARY KEY (deviceuuid, datatime);

ALTER TABLE ONLY telemetry.plant_realtime
    ADD CONSTRAINT pk_plant_realtime PRIMARY KEY (plantuuid, datatime);

ALTER TABLE ONLY telemetry.plants
    ADD CONSTRAINT pk_plants PRIMARY KEY (plantuuid);

ALTER TABLE ONLY telemetry.plantas_eventos
    ADD CONSTRAINT plantas_eventos_pkey PRIMARY KEY (plantuuid, tipo, desde);

ALTER TABLE ONLY telemetry.plantas_fv_caras
    ADD CONSTRAINT plantas_fv_caras_pkey PRIMARY KEY (plantuuid, cara);

ALTER TABLE ONLY telemetry.plantas_fv_etapas
    ADD CONSTRAINT plantas_fv_etapas_pkey PRIMARY KEY (plantuuid, desde);

ALTER TABLE ONLY telemetry.plantas_fv
    ADD CONSTRAINT plantas_fv_pkey PRIMARY KEY (plantuuid);

ALTER TABLE ONLY telemetry.etl_gaps
    ADD CONSTRAINT uq_etl_gaps_ventana UNIQUE (deviceuuid, inicio, fin);

CREATE INDEX ideam_observaciones_datatime_idx ON meteo.ideam_observaciones USING btree (datatime DESC);

CREATE INDEX ix_plantas_grupos_grupo ON meteo.plantas_grupos USING btree (grupo);

CREATE INDEX nasa_power_datatime_idx ON meteo.nasa_power USING btree (datatime DESC);

CREATE INDEX nsrdb_goes_datatime_idx ON meteo.nsrdb_goes USING btree (datatime DESC);

CREATE INDEX om_air_quality_datatime_idx ON meteo.om_air_quality USING btree (datatime DESC);

CREATE INDEX om_archive_datatime_idx ON meteo.om_archive USING btree (datatime DESC);

CREATE INDEX om_historical_forecast_datatime_idx ON meteo.om_historical_forecast USING btree (datatime DESC);

CREATE INDEX om_previous_runs_datatime_idx ON meteo.om_previous_runs USING btree (datatime DESC);

CREATE INDEX om_satellite_datatime_idx ON meteo.om_satellite USING btree (datatime DESC);

CREATE INDEX om_single_runs_datatime_idx ON meteo.om_single_runs USING btree (datatime DESC);

CREATE INDEX ix_devices_plantuuid ON telemetry.devices USING btree (plantuuid);

CREATE INDEX ix_etl_gaps_estado ON telemetry.etl_gaps USING btree (estado, creacion);

CREATE INDEX ix_etl_incidencias_deviceuuid ON telemetry.etl_incidencias USING btree (deviceuuid, iniciotime DESC) WHERE (deviceuuid IS NOT NULL);

CREATE INDEX ix_etl_incidencias_plantuuid ON telemetry.etl_incidencias USING btree (plantuuid, iniciotime DESC) WHERE (plantuuid IS NOT NULL);

CREATE INDEX ix_plants_cityid ON telemetry.plants USING btree (cityid);

ALTER TABLE ONLY meteo.bloques_no_disponibles
    ADD CONSTRAINT bloques_no_disponibles_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.celdas
    ADD CONSTRAINT celdas_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.celdas
    ADD CONSTRAINT fk_celdas_modelo FOREIGN KEY (fuente, modelo) REFERENCES meteo.modelos(fuente, modelo);

ALTER TABLE ONLY meteo.ideam_grupos_estaciones
    ADD CONSTRAINT ideam_grupos_estaciones_codigoestacion_fkey FOREIGN KEY (codigoestacion) REFERENCES meteo.ideam_estaciones(codigoestacion);

ALTER TABLE ONLY meteo.ideam_grupos_estaciones
    ADD CONSTRAINT ideam_grupos_estaciones_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.modelos
    ADD CONSTRAINT modelos_fuente_fkey FOREIGN KEY (fuente) REFERENCES meteo.fuentes(fuente);

ALTER TABLE ONLY meteo.nasa_power
    ADD CONSTRAINT nasa_power_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.nsrdb_goes
    ADD CONSTRAINT nsrdb_goes_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.om_air_quality
    ADD CONSTRAINT om_air_quality_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.om_archive
    ADD CONSTRAINT om_archive_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.om_historical_forecast
    ADD CONSTRAINT om_historical_forecast_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.om_previous_runs
    ADD CONSTRAINT om_previous_runs_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.om_satellite
    ADD CONSTRAINT om_satellite_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.om_single_runs
    ADD CONSTRAINT om_single_runs_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.plantas_grupos
    ADD CONSTRAINT plantas_grupos_grupo_fkey FOREIGN KEY (grupo) REFERENCES meteo.grupos(grupo);

ALTER TABLE ONLY meteo.plantas_grupos
    ADD CONSTRAINT plantas_grupos_plantuuid_fkey FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid);

ALTER TABLE ONLY meteo.variables
    ADD CONSTRAINT variables_fuente_fkey FOREIGN KEY (fuente) REFERENCES meteo.fuentes(fuente);

ALTER TABLE ONLY telemetry.devices_fv
    ADD CONSTRAINT devices_fv_deviceuuid_fkey FOREIGN KEY (deviceuuid) REFERENCES telemetry.devices(deviceuuid);

ALTER TABLE ONLY telemetry.devices
    ADD CONSTRAINT fk_devices_devicetype FOREIGN KEY (devicetype) REFERENCES telemetry.device_types(devicetype);

ALTER TABLE ONLY telemetry.devices
    ADD CONSTRAINT fk_devices_plantuuid FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid);

ALTER TABLE ONLY telemetry.etl_ciclos
    ADD CONSTRAINT fk_etl_ciclos_servicio FOREIGN KEY (servicio) REFERENCES telemetry.etl_servicios(servicio);

ALTER TABLE ONLY telemetry.etl_gaps
    ADD CONSTRAINT fk_etl_gaps_deviceuuid FOREIGN KEY (deviceuuid) REFERENCES telemetry.devices(deviceuuid) ON DELETE CASCADE;

ALTER TABLE ONLY telemetry.etl_incidencias
    ADD CONSTRAINT fk_etl_incidencias_servicio FOREIGN KEY (servicio) REFERENCES telemetry.etl_servicios(servicio);

ALTER TABLE ONLY telemetry.inverter_history
    ADD CONSTRAINT fk_inverter_history_deviceuuid FOREIGN KEY (deviceuuid) REFERENCES telemetry.devices(deviceuuid);

ALTER TABLE ONLY telemetry.inverter_realtime
    ADD CONSTRAINT fk_inverter_realtime_deviceuuid FOREIGN KEY (deviceuuid) REFERENCES telemetry.devices(deviceuuid);

ALTER TABLE ONLY telemetry.meter_history
    ADD CONSTRAINT fk_meter_history_deviceuuid FOREIGN KEY (deviceuuid) REFERENCES telemetry.devices(deviceuuid);

ALTER TABLE ONLY telemetry.meter_realtime
    ADD CONSTRAINT fk_meter_realtime_deviceuuid FOREIGN KEY (deviceuuid) REFERENCES telemetry.devices(deviceuuid);

ALTER TABLE ONLY telemetry.plant_realtime
    ADD CONSTRAINT fk_plant_realtime_plantuuid FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid);

ALTER TABLE ONLY telemetry.plants
    ADD CONSTRAINT fk_plants_businesstype FOREIGN KEY (businesstype) REFERENCES telemetry.business_types(businesstype);

ALTER TABLE ONLY telemetry.plants
    ADD CONSTRAINT fk_plants_cityid FOREIGN KEY (cityid) REFERENCES telemetry.cities(cityid);

ALTER TABLE ONLY telemetry.plantas_eventos
    ADD CONSTRAINT plantas_eventos_plantuuid_fkey FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid) ON DELETE CASCADE;

ALTER TABLE ONLY telemetry.plantas_fv_caras
    ADD CONSTRAINT plantas_fv_caras_plantuuid_fkey FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid) ON DELETE CASCADE;

ALTER TABLE ONLY telemetry.plantas_fv_etapas
    ADD CONSTRAINT plantas_fv_etapas_plantuuid_fkey FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid) ON DELETE CASCADE;

ALTER TABLE ONLY telemetry.plantas_fv
    ADD CONSTRAINT plantas_fv_modulo_fkey FOREIGN KEY (modulo) REFERENCES telemetry.modulos_fv(modelo);

ALTER TABLE ONLY telemetry.plantas_fv
    ADD CONSTRAINT plantas_fv_plantuuid_fkey FOREIGN KEY (plantuuid) REFERENCES telemetry.plants(plantuuid) ON DELETE CASCADE;
