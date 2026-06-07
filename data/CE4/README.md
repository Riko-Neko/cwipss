# Chang'e-4 LFRS 2C Data

The Chang'e-4 lander carries the Low-Frequency Radio Spectrometer (LFRS).
Its public 2C products contain derived low-frequency spectral measurements
covering approximately 100 kHz to 40 MHz. Each PDS4 product is distributed as
a binary `.2C` data file and a paired `.2CL` XML label containing timing,
frequency-band, record-layout, and provenance metadata.

Cwipss currently reads this format. Keep each `.2C` file next to its `.2CL`
label locally so the reader can recover the frequency axis and sampling
interval. Both file types are intentionally ignored by Git.

## Public Data

- [Official Chang'e-4 data catalogue](https://moon.bao.ac.cn/ce5web/searchOrder-ce4.do)
- [Lunar and Planetary Data Release System](https://moon.bao.ac.cn/)
- [2019 LFRS 2C dataset DOI](https://doi.org/10.12350/CLPDS.GRAS.CE4.LFRS-2C-2019.vB)

The catalogue identifies the National Astronomical Observatories of China /
Ground Research and Application System as the producer. The public LFRS 2C
collection spans observations from January 3, 2019 through June 2, 2022.

## Download

### Demo data

Download one small observation and its label (about 6 MB) for a quick Cwipss
test.

Python script (recommended):

```bash
./data/CE4/download_ce4_lfrs_2c.py demo
```

Command line (`curl` and `jq`):

```bash
cd data/CE4
curl -sG 'https://moon.bao.ac.cn/moon-admin/client/science/dataInfoList' \
  --data-urlencode 'pageNum=1' --data-urlencode 'pageSize=1000' \
  --data-urlencode 'taskName=CE4' --data-urlencode 'loadName=LFRS' \
  --data-urlencode 'levelName=2C' |
jq -r '.rows[] | select(.name | contains("20190103162200_20190103174300_0001_B.")) | .dataInfoId' |
while read -r id; do
  url=$(curl -s "https://moon.bao.ac.cn/moon-admin/client/science/dataInfo/getAnnexZip/$id" |
    jq -r '.data')
  curl -L -O -C - "$url"
done
```

### Complete data

Download the complete public LFRS 2C collection, including both `.2C` data
files and `.2CL` labels.

Python script (recommended):

```bash
./data/CE4/download_ce4_lfrs_2c.py
```

Command line (`curl` and `jq`):

```bash
cd data/CE4
curl -sG 'https://moon.bao.ac.cn/moon-admin/client/science/dataInfoList' \
  --data-urlencode 'pageNum=1' \
  --data-urlencode 'pageSize=1000' \
  --data-urlencode 'taskName=CE4' \
  --data-urlencode 'loadName=LFRS' \
  --data-urlencode 'levelName=2C' |
jq -r '.rows[].dataInfoId' |
while read -r id; do
  url=$(curl -s "https://moon.bao.ac.cn/moon-admin/client/science/dataInfo/getAnnexZip/$id" |
    jq -r '.data')
  curl -L -O -C - "$url"
done
```
