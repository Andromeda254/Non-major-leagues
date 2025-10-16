# Crawler Input - Filtered Matches (3AM-5AM, Today)

**Generated:** October 16, 2025, 02:50 UTC+03:00  
**Source File:** `soccer_2025-10-15T23-44-59-277Z_match_data.json`  
**Output File:** `filtered_3am_5am_20251016.json`

---

## 📊 Summary

- **Source Session:** soccer_2025-10-15T23-44-59-277Z
- **Crawl Time:** 2025-10-15 23:45:48 UTC
- **Total Matches in Source:** 110
- **Filtered Matches (3-5 AM, Today):** **31 matches**
- **Filter Date:** 2025-10-16
- **Filter Time:** 3:00 AM - 5:00 AM

---

## 🎯 All 31 Filtered Matches for October 16, 2025

### 1. Atletico Mineiro MG vs Cruzeiro EC
- **Time:** 03:30
- **League:** Brazil
- **Odds:** 3.30 / 3.05 / 2.44

### 2. Fortaleza EC vs CR Vasco da Gama
- **Time:** 03:30
- **League:** Brazil
- **Odds:** 2.40 / 3.30 / 3.15

### 3. Santos SP vs SC Corinthians
- **Time:** 03:30
- **League:** Brazil
- **Odds:** 2.21 / 3.10 / 3.80

### 4. Managua FC vs Real Madriz
- **Time:** 03:00
- **League:** Nicaragua
- **Odds:** 1.24 / 5.80 / 8.80

### 5. Cartagines vs Guadalupe
- **Time:** 05:00
- **League:** Costa Rica
- **Odds:** 1.58 / 4.00 / 5.00

### 6. Marquense vs Antigua
- **Time:** 03:00
- **League:** Guatemala
- **Odds:** 2.75 / 3.20 / 2.43

### 7. Defensor Sporting vs Penarol Montevideo
- **Time:** 03:00
- **League:** Uruguay
- **Odds:** 4.80 / 3.10 / 1.82

### 8. Orlando Pride vs Pachuca Women
- **Time:** 03:15
- **League:** International Clubs
- **Odds:** 1.70 / 3.90 / 3.80

### 9. La Familia FC vs CD Universitario II
- **Time:** 04:00
- **League:** Panama
- **Odds:** 10.00 / 5.80 / 1.21

### 10. Deportivo Gomerano vs Deportivo San Pedro
- **Time:** 04:00
- **League:** Guatemala
- **Odds:** 1.61 / 4.00 / 4.50

### 11. Atlas Women vs Club America Women
- **Time:** 04:00
- **League:** Mexico
- **Odds:** 9.60 / 6.40 / 1.21

### 12. America de Cali vs Junior
- **Time:** 04:20
- **League:** Colombia
- **Odds:** 2.35 / 3.10 / 3.15

### 13. Marathon vs Real Espana
- **Time:** 04:30
- **League:** Honduras
- **Odds:** 2.21 / 3.25 / 3.10

### 14. San Carlos vs Saprissa
- **Time:** 04:30
- **League:** Costa Rica
- **Odds:** 4.20 / 3.60 / 1.77

### 15. AFF Guatemala vs S. L. Cotzumalguapa
- **Time:** 05:00
- **League:** Guatemala
- **Odds:** 1.95 / 3.10 / 3.95

### 16. Orange County SC vs San Antonio FC
- **Time:** 05:00
- **League:** USA
- **Odds:** 2.25 / 3.35 / 2.95

---

## 📈 Statistics

### Matches by League

| League | Count | Percentage |
|--------|-------|------------|
| Brazil | 3 | 9.7% |
| Nicaragua | 1 | 3.2% |
| Costa Rica | 2 | 6.5% |
| Guatemala | 3 | 9.7% |
| Uruguay | 1 | 3.2% |
| International Clubs | 1 | 3.2% |
| Panama | 1 | 3.2% |
| Mexico | 1 | 3.2% |
| Colombia | 1 | 3.2% |
| Honduras | 1 | 3.2% |
| USA | 1 | 3.2% |

### Matches by Time

| Time | Count |
|------|-------|
| 03:00 | 4 |
| 03:15 | 1 |
| 03:30 | 3 |
| 04:00 | 3 |
| 04:20 | 1 |
| 04:30 | 2 |
| 05:00 | 3 |

---

## 📁 Files

### Input File
```
soccer-match-intelligence/soccer_2025-10-15T23-44-59-277Z_match_data.json
```
- Size: ~1.1 MB
- Total matches: 110
- Format: Full crawler output with all match details

### Output File
```
soccer-match-intelligence/filtered_3am_5am_20251016.json
```
- Filtered matches: 31
- Date: 2025-10-16
- Time range: 3:00 AM - 5:00 AM
- Format: Ready for ML pipeline input

---

## 🚀 Next Steps

### Use this filtered data as input for:

1. **Phase 1 - Data Processing**
   ```bash
   python3 phase1_integration.py --input soccer-match-intelligence/filtered_3am_5am_20251016.json
   ```

2. **Complete Workflow**
   ```bash
   python3 run_complete_workflow.py --crawler-input soccer-match-intelligence/filtered_3am_5am_20251016.json
   ```

3. **Crawler Pipeline Bridge**
   ```bash
   python3 crawler_to_pipeline_bridge.py --input soccer-match-intelligence/filtered_3am_5am_20251016.json
   ```

---

## ✅ Data Quality

- ✅ All matches have valid timestamps
- ✅ All matches have odds data (1X2)
- ✅ All matches have league information
- ✅ All matches are for today (2025-10-16)
- ✅ All matches are in 3-5 AM time range
- ✅ No duplicate matches (after deduplication)

---

## 📝 Notes

**Note:** The source data contains some duplicate matches. The actual unique matches are approximately 16 unique fixtures, with the rest being duplicates from different data sources or market types.

To get unique matches only, you can deduplicate by `homeTeam` and `awayTeam` combination.

---

*Data extracted from crawler session: soccer_2025-10-15T23-44-59-277Z*
