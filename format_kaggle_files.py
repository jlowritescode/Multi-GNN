import numpy as np
import pandas as pd
from datetime import datetime
#from datatable import f,join,sort
import sys
import os

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

raw = pd.read_csv(inPath, low_memory=False)
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
Payment Format,Is Laundering\n"

firstTs = -1
print(raw.shape)
print(len(raw))
raw = raw.reset_index(drop=True)

with open(outPath, 'w') as writer:
    writer.write(header)

    for i, row in raw.iterrows():
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

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % (
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
        )

        writer.write(line)
formatted = pd.read_csv(outPath, low_memory=False)
#dt.fread(outPath)
formatted = formatted.sort_values(by=formatted.columns[3]).reset_index(drop=True)
formatted.to_csv(outPath)#