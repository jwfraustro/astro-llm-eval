# Gaia DR3 access notes

## ESA Gaia Archive (canonical)

- TAP base URL: `https://gea.esac.esa.int/tap-server/tap`
- Main table: `gaiadr3.gaia_source` (columns include `source_id`, `ra`, `dec`)
- ADQL uses `SELECT TOP n ...`, not `LIMIT`.
- Cone search:
  `WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', <ra>, <dec>, <radius_deg>)) = 1`
- astroquery:
  ```python
  from astroquery.gaia import Gaia
  job = Gaia.launch_job("SELECT TOP 10 source_id, ra, dec FROM gaiadr3.gaia_source WHERE ...")
  table = job.get_results()
  ```

## MAST mirror

- TAP base URL: `https://mast.stsci.edu/vo-tap/api/v0.1/gaiadr3`
- Main table: `dbo.gaia_source`
- Generic TAP access with pyvo:
  ```python
  import pyvo
  svc = pyvo.dal.TAPService("https://mast.stsci.edu/vo-tap/api/v0.1/gaiadr3")
  table = svc.search("SELECT TOP 10 source_id, ra, dec FROM dbo.gaia_source WHERE ...").to_table()
  ```

## Writing the output

`source_id` is a 64-bit integer. Write it as an integer column; converting it
through a float loses precision and produces ids that do not exist.
