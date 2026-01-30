# ---- Extract PNB Data from Supreme Court Dataset ----
# Uses the actual bank-level data with heterogeneous margins
rm(list = ls())
library(dplyr)

# Paths
output_csv <- "d:/Projects/mergerBayes/data/pnb_case_data.csv"

# Load Data
load("d:/Projects/mergerBayes/data/supreme_data.RData")
load("d:/Projects/mergerBayes/data/event_overlaps.RData")

# Get PNB event info
pnb_info <- events[events$abb == "pnb", ]
cat("PNB Case Info:\n")
print(pnb_info)

# Filter for PNB (event_id = 1) and DATA YEAR (1960 per events table)
pnb_data <- simdata %>%
  filter(event_id == 1) %>%
  filter(year == 1960) %>%  # Data year from events table
  filter(!is.na(margin) & margin > 0) %>%
  droplevels()

cat("\nFiltered PNB Data (Year 1960):\n")
cat("- Rows:", nrow(pnb_data), "\n")
cat("- Years:", unique(as.numeric(as.character(pnb_data$year))), "\n")
cat("- Banks:", length(unique(pnb_data$tophold)), "\n")
cat("- Margin range:", range(pnb_data$margin), "\n")
cat("- Rate range:", range(pnb_data$rate_deposits), "\n")
cat("- Share range:", range(pnb_data$shareIn), "\n")

# Select and rename columns to match expected format
pnb_export <- pnb_data %>%
  select(
    event_id,
    tophold,
    year,
    shareIn,
    margin,
    rate_deposits,
    total_deposits,
    total_assets
  )

# Save
write.csv(pnb_export, output_csv, row.names = FALSE)
cat("\nData saved to:", output_csv, "\n")
cat("\nPreview:\n")
print(head(pnb_export, 6))
