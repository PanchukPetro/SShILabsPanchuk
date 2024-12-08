#Variant 2
# Outlook = Overcast
# Humidity = High
# Wind = Strong

#Probability tables:
WeatherTable = [
    [2/9,3/5,5/14], #Sunny
    [4/5,0/5,4/14], #Overcast
    [3/9,2/5,5/14], #Rainy
    [9/14],[5/14]   #Total
]
HumidityTable = [
    [3/9,4/5,7/14], #High
    [6/9,1/5,7/14], #Low
    [9/14,5/14]     #Total
]
WindTable = [
    [6/9,2/5,8/14], #High
    [3/9,3/5,7/14], #Low
    [9/14,5/14]     #Total
]
YesTotal = 9/14
NoTotal = 5/14

CurrentWeather = 1
CurrentHumidity = 0
CurrentWind = 0

YesProbability = WeatherTable[CurrentWeather][0] * HumidityTable[CurrentHumidity][0] * WindTable[CurrentWind][0] * YesTotal
NoProbability = WeatherTable[CurrentWeather][1] * HumidityTable[CurrentHumidity][1] * WindTable[CurrentWind][1] * NoTotal

NormalizedYesProbability = YesProbability / (YesProbability + NoProbability)
NormalizedNoProbability = 1-NormalizedYesProbability

print("Вірогітність проведення гри при даних умовах:")
print(round(NormalizedYesProbability*100,2),"%")
print("Вірогітність що гру не проведуть:")
print(round(NormalizedNoProbability*100,2),"%")


