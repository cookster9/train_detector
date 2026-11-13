BEGIN;

-- Add a Google Maps link for each crossing.
ALTER TABLE public.crossings
ADD COLUMN google_maps_url TEXT;

-- Set the Google Maps link for this crossing.
UPDATE public.crossings
SET google_maps_url = 'https://www.google.com/maps/place/36.1935047,-86.7550428'
WHERE fra_crossing_id = '350175T';

COMMIT;