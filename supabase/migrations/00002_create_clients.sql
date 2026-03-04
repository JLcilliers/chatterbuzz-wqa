-- Clients and related configuration tables

CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    domain TEXT NOT NULL,
    industry TEXT NOT NULL DEFAULT '',
    business_type TEXT NOT NULL DEFAULT '',
    target_locations TEXT[] DEFAULT '{}',
    goals JSONB DEFAULT '{}',
    brand_voice JSONB DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'onboarding'
        CHECK (status IN ('onboarding', 'active', 'paused', 'churned')),
    owner_user_id UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE client_api_credentials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    provider TEXT NOT NULL
        CHECK (provider IN ('google', 'wordpress', 'webflow', 'asana', 'gbp')),
    credentials_encrypted TEXT NOT NULL,  -- encrypted with pgcrypto
    scopes TEXT[] DEFAULT '{}',
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (client_id, provider)
);

CREATE TABLE business_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    rule_type TEXT NOT NULL,
    rule_config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER clients_updated_at
    BEFORE UPDATE ON clients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
