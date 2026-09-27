# Mostransport live-demo fix

1. Replace `backend/src/prediction/trip-matcher.service.ts` with the file from this package.
2. Add `database/03-demo-vehicles.sql` to the PostgreSQL init scripts in `docker-compose.yml`:

```yaml
- ./database/03-demo-vehicles.sql:/docker-entrypoint-initdb.d/03-demo-vehicles.sql:ro
```

Important: Docker init scripts run only when the PostgreSQL data volume is created. The current database already has the mapping, so for the current demo only the TripMatcher code change is required. The SQL file makes a fresh checkout reproducible.

The change keeps the required 10–15 minute prediction horizon. It only skips the 1 km GPS geometry check when `vehicles.current_tr_id` is explicitly assigned. Vehicles without a trusted trip mapping still use the original 1 km spatial matching.
