from abc import ABCMeta, abstractstaticmethod
from collections import deque
import threading
import time
from time import sleep
from datetime import datetime, timedelta
import requests
import json
import hmac
import hashlib
import numpy as numpy
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygad.torchga
import pygad
import pandas as pd
import sys
from matplotlib import pyplot as plt

sys.setrecursionlimit(10**9)
threading.stack_size(10**8)

startSliceTraining = 8160
endSliceTraining = 8256

currentCoinIndex = None

trainingData = {}
trainingPrices = {}
NeuralNetworks = {}

def trainModels(coin):
    global currentCoinIndex, startSliceTraining, endSliceTraining
    torch_ga = pygad.torchga.TorchGA(model=NeuralNetworks[coin.name], num_solutions=10)
    
    currentCoinIndex = coin.name

    while endSliceTraining < len(trainingData[currentCoinIndex]) - 400:

        ga_instance = pygad.GA(num_generations=100,
                            num_parents_mating=5,
                            initial_population=torch_ga.population_weights,
                            fitness_func=fitness_func,
                            mutation_type="scramble")
        
        ga_instance.run()

        ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        best_solution_weights = pygad.torchga.model_weights_as_dict(model=NeuralNetworks[coin.name], weights_vector=solution)
        
        NeuralNetworks[coin.name].load_state_dict(best_solution_weights)

        startSliceTraining += 1 
        endSliceTraining += 1

        print(endSliceTraining)
    
    investment = 100
    investmentHistory = []
    for index in range(len(trainingData[currentCoinIndex]) - 400, len(trainingData[currentCoinIndex])):

        price = trainingPrices[currentCoinIndex][index]
        stats = trainingData[currentCoinIndex][index]

        action = [0,0,0]
        stats = torch.tensor(stats, dtype=torch.float)
        predictions = pygad.torchga.predict(model=NeuralNetworks[currentCoinIndex], solution=solution, data=stats)
        move = torch.argmax(predictions).item()
        action[move] = 1

        profitPercentage = 0
        buys = []

        if action == [1,0,0]:
            buys.append(price)
            investmentHistory.append(investment)

        elif action == [0,1,0]:
            if len(buys) != 0:
                investment = investment + ((price - (sum(buys) / len(buys))) / (sum(buys) / len(buys))) * investment
                
                buys.clear()
            
            investmentHistory.append(investment)

        elif action == [0,0,1]:
            investmentHistory.append(investment)
            pass

    startSliceTraining = 0
    endSliceTraining = 96
    
    plt.plot(investmentHistory)
    plt.show() 

def fitness_func(solution, solution_idx):
    global currentCoinIndex, startSliceTraining, endSliceTraining
    
    priceArray = numpy.array(trainingPrices[currentCoinIndex])
    stats = numpy.array(trainingData[currentCoinIndex])

    priceSlice = priceArray[startSliceTraining:endSliceTraining]
    statsSlice = stats[startSliceTraining:endSliceTraining]

    for price, stats in zip(priceSlice, statsSlice):
        action = [0,0,0]
        stats = torch.tensor(stats, dtype=torch.float)
        prediction = pygad.torchga.predict(model=NeuralNetworks[currentCoinIndex], solution=solution, data=stats)
        move = torch.argmax(prediction).item()
        action[move] = 1

        profitPercentage = 0
        buys = []
        if action == [1,0,0]:
            buys.append(price)
        elif action == [0,1,0]:
            if len(buys) != 0:
                profitPercentage += ((price - (sum(buys) / len(buys))) / (sum(buys) / len(buys))) * 100
                buys.clear()
            else:
                pass

        elif action == [0,0,1]:
            pass
        
    return profitPercentage

def ReadableTimeToTimeStamp(readableTime):
        s = datetime.strptime(readableTime, "%Y-%m-%d %H:%M:%S.%f")
        timestamp = datetime.timestamp(s)
        timestamp = int(timestamp * 1000)
        return timestamp

def TimeStampToReadableTime(timeStamp):
    readableTime = datetime.fromtimestamp(timeStamp/1000)
    return readableTime

class User:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class Coins:
    def __init__(self):
        self.coin_list = []

    def startUp(self):
        while True:
            coinName = input("What will you be investing in?: ")
            self.coin_list.append(Coin(f"{coinName}"))
            more = input("more? y/n: ")
            if more == 'n':
                break
            elif more == 'y':
                continue
            else:
                print("invalid input")
                break

    def addCoin(self, name):
        pass

    def removeCoin(self, name):
        pass

    def verifyCoin(self, name):
        timeNow = datetime.now()
        startTime = timeNow - timedelta(minutes=1)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        try:
            binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1m&startTime={start}&endTime={end}'
            data = requests.get(binanceUrl).json()
        except:
            print("Exiting...")
            print("Internet may be unstable")
            os._exit(1) 

        if isinstance(data, dict):
            return False
        else:
            return True

class Coin:
    def __init__(self, name):
        self.name = name
        self.price = None
        self.smallest_precision = None
        self.queue_days = deque()
        self.queue_hours = deque()
        self.queue_quarter_hours = deque()
        self.current14xQuarterHourRSI = None
        self.current14xHourRSI = None
        self.current14xDayRSI = None
        self.currentEMA12vsPrice = None
        self.currentEMA26vsPrice = None
        self.historicalQuarterHourPrices = deque()
        self.historicalHourPrices = deque()
        self.historicalDayPrices = deque()

    def getCurrentPrice(self):
        request = requests.get(f'https://api.crypto.com/v2/public/get-ticker?instrument_name={self.name}_USDT')
        request = json.loads(request.text)
        self.price = float(request['result']['data'][-1]['a'])

    def getSmallestPrecision(self):
        request = requests.get(f'https://api.crypto.com/v2/public/get-instruments')
        request = json.loads(request.text)
        for instrument in request['result']['instruments']:
            if instrument['instrument_name'] == f'{self.name}_USDT':
                self.smallest_precision = float(instrument['min_quantity'])

    def getCoinBalance(self):
        req = {
        "id": 11,
        "method": "private/get-account-summary",
        "api_key": user.api_key,
        "params": {
            "currency": self.name
        },
        "nonce": int(time.time() * 1000)
        }

        paramString = ""
        if "params" in req:
            for key in req['params']:
                paramString += key
                paramString += str(req['params'][key])
        
        sigPayload = req['method'] + str(req['id']) + req['api_key'] + paramString + str(req['nonce'])
        
        req['sig'] = hmac.new(
            bytes(user.api_secret, 'utf-8'),
            msg=bytes(sigPayload, 'utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

        coinBalance = requests.post("https://api.crypto.com/v2/private/get-account-summary", json=req, headers={'Content-Type':'application/json'})
        accounts = json.loads(coinBalance.text)['result']['accounts']
        if len(accounts) == 0:
            return 0
        else:
            amount = json.loads(coinBalance.text)['result']['accounts'][0]['available']
            return amount

    def fillDayQueue(self):
        timeNow = datetime.now()
        startTime = timeNow - timedelta(days = 26)
        endTime = timeNow
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=1d&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
        for row in data:
            self.queue_days.append(float(row[4]))

    def updateDayQueue(self):
        self.queue_days.popleft()
        self.queue_days.append(self.price)

    def fillHourQueue(self):
        timeNow = datetime.now()
        startTime = timeNow - timedelta(hours = 14)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=1h&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
        for row in data:
            self.queue_hours.append(float(row[4])) 

    def updateHourQueue(self):
        self.queue_hours.popleft()
        self.queue_hours.append(self.price)

    def fillQuarterHourQueue(self):
        timeNow = datetime.now()
        startTime = timeNow - timedelta(minutes = 210)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=15m&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
        for row in data:
            self.queue_quarter_hours.append(float(row[4]))

    def updateQuarterHourQueue(self):
        self.queue_quarter_hours.popleft()
        self.queue_quarter_hours.append(self.price)

    def getCurrent14xQuarterHourRSI(self):
        arrayFor14mRSI = numpy.array(self.queue_quarter_hours)
        delta14m = numpy.diff(arrayFor14mRSI)
        up14m, down14m = delta14m.copy(), delta14m.copy()
        up14m[up14m < 0] = 0
        down14m[down14m > 0] = 0
        RS14m = numpy.average(up14m) / abs(numpy.average(down14m))
        RSI14m = (100 - 100 / (1 + RS14m))
        self.current14xQuarterHourRSI = RSI14m

    def getCurrent14xHourRSI(self):
        arrayFor14hRSI = numpy.array(self.queue_hours)
        arrayFor14hRSI[-1] = self.price
        delta14h = numpy.diff(arrayFor14hRSI)
        up14h, down14h = delta14h.copy(), delta14h.copy()
        up14h[up14h < 0] = 0
        down14h[down14h > 0] = 0
        RS14h = numpy.average(up14h) / abs(numpy.average(down14h))
        RSI14h = (100 - 100 / (1 + RS14h))
        self.current14xHourRSI = RSI14h

    def getCurrent14xDayRSI(self):
        arrayFor14dRSI = numpy.array(self.queue_days)
        arrayFor14dRSI[-1] = self.price
        delta14d = numpy.diff(arrayFor14dRSI)
        up14d, down14d = delta14d.copy(), delta14d.copy()
        up14d[up14d < 0] = 0
        down14d[down14d > 0] = 0
        RS14d = numpy.average(up14d) / abs(numpy.average(down14d))
        RSI14d = (100 - 100 / (1 + RS14d))
        self.current14xDayRSI = RSI14d

    def getCurrentEMA12vsPrice(self):
        # cut the day queue so that it's just 12 days and not 26
        # make sure that the current price is at the end of the queue instead of the last candle
        priceList = list(numpy.array(self.queue_days)[14:])
        priceList[-1] = self.price
        price12dDF = pd.DataFrame(priceList)
        EMA12d = price12dDF.ewm(span=12, adjust=False).mean()
        currentEMA12d = EMA12d.iloc[-1] 
        EMA12vsPrice = float(((currentEMA12d - self.price) / self.price) * 100)
        self.currentEMA12vsPrice = float(EMA12vsPrice)

    def getCurrentEMA26vsPrice(self):
        # make sure that the current price is at the end of the queue instead of the last candle
        priceList = list(self.queue_days)
        priceList[-1] = self.price
        price26dDF = pd.DataFrame(list(self.queue_days))
        EMA26d = price26dDF.ewm(span=26, adjust=False).mean()
        currentEMA26d = EMA26d.iloc[-1]
        EMA26vsPrice = float(((currentEMA26d - self.price) / self.price) * 100)
        self.currentEMA26vsPrice = float(EMA26vsPrice)

    def getHistoricalPrices(self):
        timeNow = datetime.now()
        timeNow = timeNow.replace(microsecond=0,second=0,minute=0,hour=0)
        startTime = timeNow - timedelta(hours = 3000)
        endTime = timeNow - timedelta(hours = 2500)
        
        while startTime < timeNow:
            start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            start = ReadableTimeToTimeStamp(start)
            end = ReadableTimeToTimeStamp(end)
            binanceQuarterlUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=15m&startTime={start}&endTime={end}'
            data15m = requests.get(binanceQuarterlUrl).json()
            for row in data15m:
                self.historicalQuarterHourPrices.append(float(row[4]))

            startTime = startTime + timedelta(hours = 125)
            endTime = endTime + timedelta(hours = 125)

        startTime = timeNow - timedelta(hours = 3000)
        endTime = timeNow - timedelta(hours = 2500)

        while startTime < timeNow:
            start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            start = ReadableTimeToTimeStamp(start)
            end = ReadableTimeToTimeStamp(end)
            binanceHourlyUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=1h&startTime={start}&endTime={end}'
            data1h = requests.get(binanceHourlyUrl).json()

            for row in data1h:
                self.historicalHourPrices.append(float(row[4]))

            startTime = startTime + timedelta(hours = 500)
            endTime = endTime + timedelta(hours = 500)

        startTime = timeNow - timedelta(days = 125)
        endTime = timeNow

        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceDailyUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=1d&startTime={start}&endTime={end}'
        data1d = requests.get(binanceDailyUrl).json()

        for row in data1d:
            self.historicalDayPrices.append(float(row[4]))

    def updateHistoricalPrices(self):
        timeNow = datetime.now()
        timeNow = timeNow.replace(microsecond=0,second=0,minute=0,hour=0)
        
        for num in range(96):
            self.historicalQuarterHourPrices.popleft()
        
        for num in range(24):
            self.historicalHourPrices.popleft()

        self.historicalDayPrices.popleft()

        startTime = timeNow - timedelta(hours=24)
        endTime = timeNow - timedelta(minutes=15)
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=15m&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()

        for row in data:
            self.historicalQuarterHourPrices.append(float(row[4]))

        startTime = timeNow - timedelta(hours=24)
        endTime = timeNow - timedelta(hours=1)
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=1h&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()

        for row in data:
            self.historicalHourPrices.append(float(row[4]))
                
        startTime = timeNow - timedelta(hours=24)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.name}USDT&interval=1d&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()

        self.historicalDayPrices.append(float(data[0][4]))

    def calculateTrainingData(self):
        
        newHistoricalQuarterHourList = []
        newHistoricalHourList = []
        newHistoricalDayList = []

        RSI14mList = []
        RSI14hList = []
        RSI14dList = []

        currentPriceVSEMA12dList = []
        currentPriceVSEMA26dList = []

        newHistoricalQuarterHourList = list(self.historicalQuarterHourPrices)
        
        for price in self.historicalHourPrices:
            for i in range(4):
                newHistoricalHourList.append(price)

        for price in self.historicalDayPrices:
            for i in range(96):
                newHistoricalDayList.append(price)
        
        # For RSI14m
        startSlice = 0
        endSlice = 14
        price15MinutesArray = numpy.array(newHistoricalQuarterHourList)
        while endSlice < len(newHistoricalQuarterHourList):
            price15MinutesArraySlice = price15MinutesArray[startSlice:endSlice]
            delta14m = numpy.diff(price15MinutesArraySlice)
            up14m, down14m = delta14m.copy(), delta14m.copy()
            up14m[up14m < 0] = 0
            down14m[down14m > 0] = 0
            numpy.seterr(divide='ignore')
            RS14m = numpy.average(up14m) / abs(numpy.average(down14m))
            RSI14m = (100 - 100 / (1 + RS14m))
            RSI14mList.append(RSI14m)
            startSlice += 1
            endSlice += 1

        startSlice = 0
        endSlice = 56
        priceHoursArray = numpy.array(newHistoricalHourList)
        while endSlice < len(newHistoricalHourList):
            hours = []
            priceHoursArraySlice = priceHoursArray[startSlice:endSlice]
            firstNumber = priceHoursArraySlice[0]
            hours.append(firstNumber)
            for prices in priceHoursArraySlice:
                if firstNumber != prices:
                    hours.append(prices)
                    firstNumber = prices
                if (len(hours) > 14):
                    hoursArray = numpy.array(hours)
                    hours = hoursArray[1:]

            hours[-1] = newHistoricalQuarterHourList[endSlice - 1]
            delta14h = numpy.diff(hours)
            up14h, down14h = delta14h.copy(), delta14h.copy()
            up14h[up14h < 0] = 0
            down14h[down14h > 0] = 0
            numpy.seterr(divide='ignore')
            RS14h = numpy.average(up14h) / abs(numpy.average(down14h))
            RSI14h = (100 - 100 / (1 + RS14h))
            RSI14hList.append(RSI14h)
            startSlice += 1
            endSlice += 1

        startSlice = 0
        endSlice = 1344
        priceDaysArray = numpy.array(newHistoricalDayList)
        while endSlice < len(newHistoricalDayList):
            days = []
            priceDaysArraySlice = priceDaysArray[startSlice:endSlice]
            firstNumber = priceDaysArraySlice[0]
            days.append(firstNumber)
            for prices in priceDaysArraySlice:
                if firstNumber != prices:
                    days.append(prices)
                    firstNumber = prices
                if (len(days) > 14):
                    daysArray = numpy.array(days)
                    days = daysArray[1:]
            
            days[-1] = newHistoricalQuarterHourList[endSlice - 1]
            
            delta14d = numpy.diff(days)
            up14d, down14d = delta14d.copy(), delta14d.copy()
            up14d[up14d < 0] = 0
            down14d[down14d > 0] = 0
            numpy.seterr(divide='ignore')
            RS14d = numpy.average(up14d) / abs(numpy.average(down14d))
            RSI14d = (100 - 100 / (1 + RS14d))
            RSI14dList.append(RSI14d)
            startSlice += 1
            endSlice += 1         

        startSlice = 0
        endSlice = 2496
        priceDaysArray = numpy.array(newHistoricalDayList)
        while endSlice < len(newHistoricalDayList):
            days = []
            priceDaysArraySlice = priceDaysArray[startSlice:endSlice]
            firstNumber = priceDaysArraySlice[0]
            days.append(firstNumber)
            for prices in priceDaysArraySlice:
                if firstNumber != prices:
                    days.append(prices)
                    firstNumber = prices
                if (len(days) > 26):
                    daysArray = numpy.array(days)
                    days = daysArray[1:]

            days[-1] = newHistoricalQuarterHourList[endSlice - 1]
            
            price12dDF = pd.DataFrame(list(days[14:]))
            price26dDF = pd.DataFrame(list(days))
            
            EMA12d = price12dDF.ewm(span=12, adjust=False).mean()
            EMA26d = price26dDF.ewm(span=26, adjust=False).mean()

            currentEMA12d = EMA12d.iloc[-1] 
            currentEMA26d = EMA26d.iloc[-1]

            EMA12vsPrice = float(((currentEMA12d - days[-1]) / days[-1]) * 100)
            EMA26vsPrice = float(((currentEMA26d - days[-1]) / days[-1]) * 100)

            currentPriceVSEMA12dList.append(EMA12vsPrice)
            currentPriceVSEMA26dList.append(EMA26vsPrice)
            
            startSlice += 1
            endSlice += 1

        trainingInfoList = [RSI14mList,RSI14hList,RSI14dList,currentPriceVSEMA12dList,currentPriceVSEMA26dList]
        trainingInfo = [list(i) for i in zip(*trainingInfoList)]
        trainingData[self.name] = trainingInfo
        trainingPrices[self.name] = list(numpy.array(newHistoricalQuarterHourList)[2496:])

class ICommand(metaclass=ABCMeta):

    @abstractstaticmethod
    def execute(): """ static method """

class GetCurrentPriceCommand(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.getCurrentPrice()

class UpdateQuarterHourQueueCommand(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.updateQuarterHourQueue()

class UpdateHourQueueCommand(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.updateHourQueue()

class UpdateDayQueueCommand(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.updateDayQueue()

class GetCurrent14xQuarterHourRSI(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.getCurrent14xQuarterHourRSI()

class GetCurrent14xHourRSI(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.getCurrent14xHourRSI()

class GetCurrent24xDayRSI(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.getCurrent14xDayRSI()

class GetCurrentEMA12vsPrice(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.getCurrentEMA12vsPrice()

class GetCurrentEMA26vsPrice(ICommand):
    def __init__(self, coin):
        self._coin = coin

    def execute(self):
        self._coin.getCurrentEMA26vsPrice()

class Timer:
    def __init__(self, timeInterval, commandList):
        self.timeInterval = timeInterval
        self._commands = commandList

    def timeLoop(self):
        threading.Thread(target=self.timeLoopThread).start()

    def timeLoopThread(self):
        timer = datetime.now()
        seconds = timer.second
        minutes = timer.minute
        hours = timer.hour
        currentSeconds = hours * 3600 + minutes * 60 + seconds
        nextSecond = ((int(currentSeconds / self.timeInterval)) * self.timeInterval) + self.timeInterval
        
        if nextSecond > 86399:
            nextSecond = 0
        
        while True:
            timer = datetime.now()
            try:
                timer = datetime.strptime(str(timer), "%Y-%m-%d %H:%M:%S.%f")
            except:
                pass

            seconds = timer.second
            minutes = timer.minute
            hours = timer.hour
            totalSecondsInDay = hours * 3600 + minutes * 60 + seconds

            if totalSecondsInDay == nextSecond:
                nextSecond += self.timeInterval
                for commands in self._commands:
                    commands.execute()
                if nextSecond > 86399:
                    nextSecond = 0
                sleep(1)
    
if  __name__ == "__main__":

    user = User(f"{input('What is your api key? ')}",f"{input('What is your api secret? ')}")

    coins = Coins()
    coins.startUp()

    for coin in coins.coin_list:
        coin.getSmallestPrecision()
    
    for coin in coins.coin_list:
        coin.getHistoricalPrices()
    
    for coin in coins.coin_list:
        coin.calculateTrainingData()
    
    for coin in coins.coin_list:
        coin.fillDayQueue()
        coin.fillHourQueue()
        coin.fillQuarterHourQueue()

        NeuralNetworks[coin.name] = NN(5, 5, 3)

    for coin in coins.coin_list:    
        trainModels(coin)

    for coin in coins.coin_list:
        GETCURRENTPRICECOMMAND = GetCurrentPriceCommand(coin)
        UPDATEQUARTERHOURQUEUECOMMAND = UpdateQuarterHourQueueCommand(coin)
        UPDATEHOURQUEUECOMMAND = UpdateHourQueueCommand(coin)
        UPDATEDAYQUEUECOMMAND = UpdateDayQueueCommand(coin)
        GETCURRENT14XQUARTERHOURRSI = GetCurrent14xQuarterHourRSI(coin)
        GETCURRENT14XHOURRSI = GetCurrent14xHourRSI(coin)
        GETCURRENT14XDAYRSI = GetCurrent24xDayRSI(coin)
        GETCURRENTEMA12VSPRICE = GetCurrentEMA12vsPrice(coin)
        GETCURRENTEMA26VSPRICE = GetCurrentEMA26vsPrice(coin)

        # instead of attaching a seperate timer to every command, you can also place several commands for one timer so that there arent too many threads
        timer1 = Timer(900, [GETCURRENTPRICECOMMAND, 
                             UPDATEQUARTERHOURQUEUECOMMAND, 
                             GETCURRENT14XQUARTERHOURRSI, 
                             GETCURRENT14XHOURRSI, 
                             GETCURRENT14XDAYRSI,
                             GETCURRENTEMA12VSPRICE,
                             GETCURRENTEMA26VSPRICE]) # attach timer to the command so it activates at certain intervals
        
        timer2 = Timer(3600, [UPDATEHOURQUEUECOMMAND])   

        timer3 = Timer(86400, [UPDATEDAYQUEUECOMMAND]) 
        timer1.timeLoop()       
        timer2.timeLoop()
        timer3.timeLoop()

# TODO
# create a class for commands. Maybe call it interface.
# Fix the time loop so that there is not a 1 second pause when the loop is done
