-- ==============================================================================
-- Follica Supabase Storage & Database Security Policy (Reference Template)
-- ==============================================================================
-- Note: Reference design template only. No application-level encryption at rest
-- is implemented; production deployments requiring encryption should leverage
-- OS-level full-disk encryption (BitLocker/LUKS) or managed encrypted cloud volumes.
--
-- Features:
-- 1. Private access-controlled storage bucket configuration
-- 2. Strict Row Level Security (RLS) policies (Users only access their own scans)
-- 3. Automated 30-Day Retention Policy Cleanup Function (Staying within 1GB Free Tier)
-- ==============================================================================

-- 1. Create Private Storage Bucket for Scalp Scans
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'follica-scalp-scans',
    'follica-scalp-scans',
    false,
    10485760, -- 10MB limit per photo
    ARRAY['image/jpeg', 'image/png', 'image/webp']
)
ON CONFLICT (id) DO NOTHING;

-- 2. Storage Bucket Row-Level Security (RLS)
CREATE POLICY "Users can upload only to their own folder"
ON storage.objects
FOR INSERT
WITH CHECK (
    bucket_id = 'follica-scalp-scans'
    AND auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can view only their own uploaded scans"
ON storage.objects
FOR SELECT
USING (
    bucket_id = 'follica-scalp-scans'
    AND auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can delete their own scans"
ON storage.objects
FOR DELETE
USING (
    bucket_id = 'follica-scalp-scans'
    AND auth.uid()::text = (storage.foldername(name))[1]
);

-- 3. User Consent & Retention Registry Table
CREATE TABLE IF NOT EXISTS public.user_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    biometric_image_consent BOOLEAN NOT NULL DEFAULT true,
    research_opt_in BOOLEAN NOT NULL DEFAULT false,
    retention_days INT NOT NULL DEFAULT 30,
    agreed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.user_consents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view and edit their own consent record"
ON public.user_consents
FOR ALL
USING (auth.uid() = user_id);

-- 4. Automated 30-Day Retention Purge Trigger / Scheduled Function
CREATE OR REPLACE FUNCTION purge_expired_scalp_scans()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Delete storage records older than user retention window (default 30 days)
    DELETE FROM storage.objects
    WHERE bucket_id = 'follica-scalp-scans'
      AND created_at < NOW() - INTERVAL '30 days';
END;
$$;
