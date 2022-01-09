import math
from datetime import date
from urllib.parse import urlparse
import pandas as pd
import whois
from pandas.core.frame import DataFrame
from tld import get_tld


def main():
    # Reading initial dataset into Pandas DataFrame
    initialData = pd.read_csv("./data/initialData.csv", sep=",")
    print(initialData)
    newColumns = ["length", "numAlphas", "numDigits", "alphaDigitRatio", "entropy", "isHTTP", "isHTTPS", "params", "anchors", "directories", "tld", "age", "class"]
    dataRows = []
    for index, row in initialData.iterrows():
        curRow = []
        curRow.append(getLength(row[0]))

        alphas = alphaCount(row[0])
        digits = digitCount(row[0])
        curRow.append(alphas)
        curRow.append(digits)

        if digits != 0:
            curRow.append(alphas/digits)
        else:
            curRow.append(1)
        curRow.append(URLentropy(row[0]))
        curRow.append(isHTTP(row[0]))
        curRow.append(isHTTPS(row[0]))
        curRow.append(parameterCount(row[0]))
        curRow.append(anchorCount(row[0]))
        curRow.append(directoryCount(row[0]))

        tld = getTLD(row[0])

        if tld is not None:
            curRow.append(tld)
            age = getAge(row[0])
            if age is not None:
                curRow.append(age)
                curRow.append(row[1])
                dataRows.append(curRow)

    newData = DataFrame(dataRows, columns=newColumns)
    newData.to_csv("./data/urls.data", index=False, header=True)

# Helper functions for extracting URL string features
def getLength(row):
    return len(row)

def digitCount(row):
        digits = [i for i in row if i.isdigit()]
        return len(digits)

def URLentropy(row):
    # Calculating the Shannon entropy of the URL

    # Calculating probability of characters in URL
    charProb = [ float(row.count(c)) / len(row) for c in dict.fromkeys(list(row)) ]

    # Calculating the entropy of the URL
    stringEntropy = - sum([ p * math.log(p) / math.log(2.0) for p in charProb ])

    return stringEntropy

def isHTTP(row):
    if "http:" in row:
        return 1
    else:
        return 0

def isHTTPS(row):
    if "https:" in row:
        return 1
    else:
        return 0

def parameterCount(row):
        if "?" in row:
            if "&" in row:
                parameters = row.split('&')
            else:
                return 1
            return len(parameters)
        return 0

def anchorCount(row):
    anchorCount = row.split('#')
    return len(anchorCount) - 1

def directoryCount(row):
    directoryCount = row.split('http')[-1].split('//')[-1].split('/')
    return len(directoryCount)-1

def digitCount(row):
    digits = 0
    for character in row:
        if character.isdigit():
            digits = digits + 1
    return digits

def alphaCount(row):
    alphas = 0
    for character in row:
        if character.isalpha():
            alphas = alphas + 1
    return alphas

def getTLD(row):
    try:
        tld = get_tld(row)
    except:
        return None
    return tld

def getAge(row):
    domain = urlparse(row).netloc

    try:
        w = whois.whois(domain)
        if isinstance(w.creation_date, list):
            delta = date.today() - w.creation_date[0].date()
        else:
            delta = date.today() - w.creation_date.date()
    except:
        return None
    return delta.days

if __name__ == "__main__":
    main()
