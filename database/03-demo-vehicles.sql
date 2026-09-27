-- Demo mapping for the organizer-provided random telemetry emulator.
-- The emulator units do not carry route identity, so Backend needs an
-- authoritative vehicle -> trip mapping for the live demo.
INSERT INTO vehicles (unit_id, current_tr_id)
VALUES
  ('1001', '134040'),
  ('1002', '134040'),
  ('1003', '134040'),
  ('1004', '134040')
ON CONFLICT (unit_id) DO UPDATE
SET current_tr_id = EXCLUDED.current_tr_id;
