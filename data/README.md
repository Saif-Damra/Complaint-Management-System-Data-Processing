# Dataset Information

## About the Data

The dataset (`Complaints.csv`) contains customer complaint records from a telecom company. Due to privacy considerations, the raw data is **not included** in this repository.

## Dataset Schema

| Column | Description | Type |
|--------|-------------|------|
| `CASE_ID` | Unique identifier for each complaint case | String |
| `OFFER_NAME` | Name of the customer's subscribed offer | String |
| `CUSTOMER_TYPE` | Type of customer (e.g., CBU, EBU) | String |
| `CUSTOMER_GROUP` | Customer group classification | String |
| `CURRENT_STATUS` | Current status of the complaint (e.g., Resolved) | String |
| `ESCALATION_FLAG` | Whether the case was escalated (Yes/No) | String |
| `ESCALATED_GROUP` | Group the case was escalated to | String |
| `OPEN_DATE` | Date and time the complaint was opened | DateTime |
| `OPEN_USER` | User who opened the complaint | String |
| `CLOSE_DATE` | Date and time the complaint was closed | DateTime |
| `CLOSE_GROUP` | Group that closed the complaint | String |
| `CLOSE_USER` | User who closed the complaint | String |
| `AGE_BRACKET` | Age bracket of the complaint (in days) | Float |
| `ACTUAL_COMPLAINT` | Whether it's an actual complaint | String |
| `CALLBACK_MECHANISM` | How the customer was contacted back | String |
| `RESOLUTION` | Resolution details | String |
| `RESOLUTION_DESCRIPTION` | Detailed resolution description | String |
| `CASE_DESC` | Case description | String |
| `OPEN_GR` | Opening group | String |
| `COMPLAINT_TYPE` | **Target variable** - Type of complaint (Technical/Commercial) | String |
| `PRODUCT` | Product related to the complaint | String |
| `CASE` | Case classification | String |

## How to Use

1. Place your `Complaints.csv` file in this `data/` directory
2. Run the scripts from the project root:
   ```bash
   python src/main.py
   ```

> **Note:** The `.gitignore` is configured to exclude CSV files from this directory to protect data privacy.
