-- OHH-206: VR session registry.
--
-- Apply this file in the Ohho Supabase project (the same project OhhO-VR
-- already uses for `user_robots`). This repo does not ship the website
-- clone. Paste the file into the Supabase SQL editor and run it once.
-- Full steps: docs/sessions.md
--
-- The raw OHH-196 gateway token is never a column. `token_hash` is the
-- SHA-256 hex digest of that token. Stale rows stay in the table until
-- `ohho sim down` / `ohho robot announce --down` deletes them, and
-- `robot_sessions_visible` hides them once `expires_at` has passed.

create table if not exists public.robot_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  robot_id text not null,
  kind text not null,
  target text not null,
  rosbridge_url text not null,
  video_urls jsonb not null default '[]'::jsonb,
  webrtc_signal_url text not null default '',
  token_hash text not null,
  status text not null default 'active',
  heartbeat_at timestamptz not null default now(),
  expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint robot_sessions_kind_check
    check (kind in ('sim', 'real')),
  constraint robot_sessions_target_check
    check (target in ('local', 'cloud', 'robot')),
  constraint robot_sessions_status_check
    check (status in ('active', 'ended')),
  constraint robot_sessions_token_hash_check
    check (token_hash ~ '^[0-9a-f]{64}$'),
  constraint robot_sessions_video_urls_check
    check (jsonb_typeof(video_urls) = 'array'),
  constraint robot_sessions_owner_slot_key
    unique (user_id, robot_id, kind, target)
);

comment on table public.robot_sessions is
  'OHH-206 live sim and robot endpoints for the VR garage. Owner RLS. token_hash is SHA-256 hex of the OHH-196 gateway token.';

comment on column public.robot_sessions.token_hash is
  'SHA-256 hex of the OHH-196 gateway token (OHHO_GATEWAY_TOKEN or ~/.ohho/gateway/pairing-code). The raw token is never stored.';

comment on column public.robot_sessions.expires_at is
  'Heartbeat interval is 10s. Writers set expires_at 30s after heartbeat_at. The visible view hides rows once this timestamp has passed.';

create index if not exists robot_sessions_owner_robot_live_idx
  on public.robot_sessions (user_id, robot_id, expires_at desc)
  where status = 'active';

create or replace function public.set_robot_sessions_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists robot_sessions_set_updated_at on public.robot_sessions;
create trigger robot_sessions_set_updated_at
  before update on public.robot_sessions
  for each row
  execute function public.set_robot_sessions_updated_at();

alter table public.robot_sessions enable row level security;
alter table public.robot_sessions force row level security;

drop policy if exists robot_sessions_select_own on public.robot_sessions;
create policy robot_sessions_select_own
  on public.robot_sessions
  for select
  to authenticated
  using (auth.uid() = user_id);

drop policy if exists robot_sessions_insert_own on public.robot_sessions;
create policy robot_sessions_insert_own
  on public.robot_sessions
  for insert
  to authenticated
  with check (auth.uid() = user_id);

drop policy if exists robot_sessions_update_own on public.robot_sessions;
create policy robot_sessions_update_own
  on public.robot_sessions
  for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists robot_sessions_delete_own on public.robot_sessions;
create policy robot_sessions_delete_own
  on public.robot_sessions
  for delete
  to authenticated
  using (auth.uid() = user_id);

revoke all on table public.robot_sessions from public, anon;
grant select, insert, update, delete on table public.robot_sessions to authenticated;
grant select, insert, update, delete on table public.robot_sessions to service_role;

-- security_invoker so the view runs as the calling user and RLS still applies.
-- A security_definer view would leak every owner's sessions to the headset.
create or replace view public.robot_sessions_visible
with (security_invoker = true) as
select
  id,
  user_id,
  robot_id,
  kind,
  target,
  rosbridge_url,
  video_urls,
  webrtc_signal_url,
  token_hash,
  status,
  heartbeat_at,
  expires_at,
  created_at,
  updated_at
from public.robot_sessions
where status = 'active'
  and expires_at > now();

comment on view public.robot_sessions_visible is
  'OHH-206 sessions the VR garage may list. Stale and ended rows are hidden. RLS is the owner''s rows only.';

revoke all on table public.robot_sessions_visible from public, anon;
grant select on table public.robot_sessions_visible to authenticated;
grant select on table public.robot_sessions_visible to service_role;

revoke all on function public.set_robot_sessions_updated_at() from public;

notify pgrst, 'reload schema';
