BEGIN;

-- Allow frontend users to read crossings.
GRANT SELECT ON TABLE public.crossings
TO anon, authenticated;

COMMIT;