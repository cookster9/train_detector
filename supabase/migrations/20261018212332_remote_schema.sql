drop extension if exists "pg_net";


  create table "public"."detections" (
    "id" uuid not null default gen_random_uuid(),
    "timestamp" text not null,
    "label" text not null,
    "score" double precision not null,
    "sample_rate" integer not null,
    "audio_clip_path" text,
    "device_id" text,
    "metadata" jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
      );


alter table "public"."detections" enable row level security;

CREATE INDEX detections_created_at_idx ON public.detections USING btree (created_at DESC);

CREATE INDEX detections_label_idx ON public.detections USING btree (label);

CREATE UNIQUE INDEX detections_pkey ON public.detections USING btree (id);

CREATE INDEX detections_timestamp_idx ON public.detections USING btree ("timestamp" DESC);

alter table "public"."detections" add constraint "detections_pkey" PRIMARY KEY using index "detections_pkey";

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.get_today_crossing_count()
 RETURNS integer
 LANGUAGE plpgsql
AS $function$DECLARE
    detection_count integer;
BEGIN
    SELECT COUNT(*)
    INTO detection_count
    FROM detections
    WHERE label IN ('train_faraway', 'train_close')
      AND (created_at AT TIME ZONE 'America/Chicago')::date =
          (CURRENT_TIMESTAMP AT TIME ZONE 'America/Chicago')::date;

    RETURN detection_count;
END;$function$
;

CREATE OR REPLACE FUNCTION public.get_train_likelihood()
 RETURNS TABLE(hour_ct integer, bucket_15min integer, count bigint)
 LANGUAGE plpgsql
AS $function$
BEGIN
  RETURN QUERY
  SELECT 
    EXTRACT(HOUR FROM created_at AT TIME ZONE 'America/Chicago')::INT AS hour_ct,
    (FLOOR(EXTRACT(MINUTE FROM created_at AT TIME ZONE 'America/Chicago') / 15) * 15)::INT AS bucket_15min,
    COUNT(*)
  FROM detections
  WHERE label = 'train_close'
    AND created_at > NOW() - INTERVAL '7 days'
  GROUP BY hour_ct, bucket_15min
  ORDER BY hour_ct, bucket_15min;
END;
$function$
;

grant delete on table "public"."detections" to "anon";

grant insert on table "public"."detections" to "anon";

grant references on table "public"."detections" to "anon";

grant select on table "public"."detections" to "anon";

grant trigger on table "public"."detections" to "anon";

grant truncate on table "public"."detections" to "anon";

grant update on table "public"."detections" to "anon";

grant delete on table "public"."detections" to "authenticated";

grant insert on table "public"."detections" to "authenticated";

grant references on table "public"."detections" to "authenticated";

grant select on table "public"."detections" to "authenticated";

grant trigger on table "public"."detections" to "authenticated";

grant truncate on table "public"."detections" to "authenticated";

grant update on table "public"."detections" to "authenticated";

grant delete on table "public"."detections" to "service_role";

grant insert on table "public"."detections" to "service_role";

grant references on table "public"."detections" to "service_role";

grant select on table "public"."detections" to "service_role";

grant trigger on table "public"."detections" to "service_role";

grant truncate on table "public"."detections" to "service_role";

grant update on table "public"."detections" to "service_role";


  create policy "web read"
  on "public"."detections"
  as permissive
  for select
  to public
using (true);


drop trigger if exists "protect_bucket_control_insert" on "storage"."buckets";

drop trigger if exists "protect_bucket_control_update" on "storage"."buckets";

drop trigger if exists "protect_bucket_control_update_role" on "storage"."buckets";


