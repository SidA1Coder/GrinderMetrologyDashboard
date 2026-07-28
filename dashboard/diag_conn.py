"""Quick ODS connection diagnostic.

Run from the dashboard folder so config.py picks up the local .env:

    cd /d C:\\GrinderMetrologyDashboard\\dashboard
    "C:\\New folder\\envs\\fs50defect\\python.exe" diag_conn.py
"""

import os
import sys

print("CWD:", os.getcwd())

try:
    import config
except Exception as e:  # noqa: BLE001
    print("!! Could not import config:", e)
    sys.exit(1)

print("FS50_MET_DRIVER env:", os.getenv("FS50_MET_DRIVER"))
print("FS50_MET_SERVER env:", os.getenv("FS50_MET_SERVER"))
print("METROLOGY_DRIVER  :", getattr(config, "METROLOGY_DRIVER", "<none>"))
print("METROLOGY_SERVER  :", getattr(config, "METROLOGY_SERVER", "<none>"))
print("METROLOGY_DATABASE:", getattr(config, "METROLOGY_DATABASE", "<none>"))

try:
    conn_str = config.metrology_odbc_str()
    print("ODBC string       :", conn_str)
except Exception as e:  # noqa: BLE001
    print("!! Could not build ODBC string:", e)
    sys.exit(1)

try:
    import pyodbc

    print("Available drivers :", pyodbc.drivers())
    cn = pyodbc.connect(conn_str, timeout=5)
    row = cn.cursor().execute("SELECT 1").fetchone()
    print("CONNECT OK        :", row)
except Exception as e:  # noqa: BLE001
    print("!! CONNECT FAILED :", repr(e))
    sys.exit(2)
