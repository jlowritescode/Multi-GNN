import numpy as np
import pandas as pd
from datetime import datetime
#from datatable import f,join,sort
import sys
import os
from pathlib import Path


def load_pattern_lookup(pattern_path):
    """
    Reads HI-Small_Patterns.txt and maps each raw transaction
    line to its laundering pattern.
    """

    lookup = {}
    current_pattern = None

    if not Path(pattern_path).exists():
        print(f"WARNING: Pattern file not found: {pattern_path}")
        return lookup

    with open(pattern_path, "r", encoding="utf-8") as f:
        for raw_line in f:

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT - "):

                descriptor = line.replace(
                    "BEGIN LAUNDERING ATTEMPT - ",
                    "",
                    1
                )

                # Examples:
                # "CYCLE: Max 10 hops" -> "CYCLE"
                # "FAN-OUT: Max 16-degree..." -> "FAN-OUT"
                current_pattern = (
                    descriptor
                    .split(":", 1)[0]
                    .strip()
                    .upper()
                )

                continue

            if line.startswith("END LAUNDERING ATTEMPT"):
                current_pattern = None
                continue

            if current_pattern is not None:
                lookup[line] = current_pattern

    return lookup



n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"
# Automatically locate the companion pattern file.
#
# HI-Small_Trans.csv
#       ->
# HI-Small_Patterns.txt

input_path = Path(inPath)

pattern_filename = input_path.name.replace(
    "_Trans.csv",
    "_Patterns.txt"
)

pattern_path = input_path.parent / pattern_filename

pattern_lookup = load_pattern_lookup(
    pattern_path
)

print(
    f"Loaded {len(pattern_lookup)} "
    f"pattern-labeled transactions"
)
raw = pd.read_csv(inPath, low_memory=False)
# ---------------------------------------------------------
# Match each original transaction to its laundering pattern
# ---------------------------------------------------------

raw_patterns = []

with open(inPath, "r", encoding="utf-8") as f:

    # Skip CSV header
    next(f)

    for line in f:

        transaction_line = line.strip()

        pattern = pattern_lookup.get(
            transaction_line,
            "NONE"
        )

        raw_patterns.append(pattern)


if len(raw_patterns) != len(raw):
    raise ValueError(
        "Pattern matching produced a different number "
        "of rows than the transaction CSV."
    )


raw["Pattern"] = raw_patterns


print("\nPattern counts:")

print(
    raw.loc[
        raw["Is Laundering"] == 1,
        "Pattern"
    ].value_counts()
)
#dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()


def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering,Pattern\n"

firstTs = -1
print(raw.shape)
print(len(raw))
raw = raw.reset_index(drop=True)

with open(outPath, 'w') as writer:
    writer.write(header)

    for i, row in raw.iterrows():
        pattern = row["Pattern"]
        datetime_object = datetime.strptime(row["Timestamp"], '%Y/%m/%d %H:%M')

        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(row["Receiving Currency"], currency)
        cur2 = get_dict_val(row["Payment Currency"], currency)

        fmt = get_dict_val(row["Payment Format"], paymentFormat)

        fromAccIdStr = str(row["From Bank"]) + str(row["Account"])
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = str(row["To Bank"]) + str(row["Account.1"])
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(row["Amount Received"])
        amountPaidOrig = float(row["Amount Paid"])

        isl = int(row["Is Laundering"])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d,%s\n' % (
            i,
            fromId,
            toId,
            int(ts),
            amountPaidOrig,
            cur2,
            amountReceivedOrig,
            cur1,
            fmt,
            isl,
            pattern
        )

        writer.write(line)
formatted = pd.read_csv(outPath, low_memory=False)

#dt.fread(outPath)
formatted = formatted.sort_values(by=formatted.columns[3]).reset_index(drop=True)
formatted.to_csv(outPath, index=False)