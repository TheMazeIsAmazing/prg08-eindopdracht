//init vars
let trainData
let testData
let predictGoodDelayed = 0
let predictWrongDelayed = 0
let predictGoodOnTime = 0
let predictWrongOnTime = 0

let airportsFrom = []
let airportsTo = []
let airlines = []

const options = {
    task: 'classification',
    debug: true,
    layers: [
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            activation: 'softmax',
        },
    ]
}

const nn = ml5.neuralNetwork(options)

function loadData() {
    Papa.parse("./data/airlines_delay.csv", {
    // Papa.parse("./data/airlines_delay_2000.csv", {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: results => normalizeRawData(results.data)
    })
}

function normalizeRawData(rawData) {
    for (let entry of rawData) {
        //if it's not present in the array, place it there
        if (airportsFrom.indexOf(entry.AirportFrom) == -1) {
            airportsFrom.push(entry.AirportFrom)
        }

        if (airportsTo.indexOf(entry.AirportTo) == -1) {
            airportsTo.push(entry.AirportTo)
        }

        if (airlines.indexOf(entry.Airline) == -1) {
            airlines.push(entry.Airline)
        }
    }

    nnAddData(rawData)
}

function nnAddData(data) {
    trainData = data.slice(0, Math.floor(data.length * 0.8))
    testData = data.slice(Math.floor(data.length * 0.8) + 1)

    for (let flight of trainData) {
        let data = {
            time: flight.Time,
            lenght: flight.Length,
            dayOfWeek: flight.DayOfWeek,
            airportFrom: airportsFrom.indexOf(flight.AirportFrom),
            airportTo: airportsTo.indexOf(flight.AirportTo),
            airline: airlines.indexOf(flight.Airline)
        }

        let label

        if (parseInt(flight.Class) === 0) {
            label = { class: "OnTime" }
        } else {
            label = { class: "Delayed" }
        }        
    
        nn.addData(data, label)
    }

    console.log(nn.neuralNetworkData.data.raw);
    nn.normalizeData()

    startTraining()
}

function startTraining() {
    nn.train({ epochs: 50 }, () => finishedTraining())
}

async function finishedTraining() {
    console.log("Finished training!")

    document.getElementById('downloadButton').addEventListener('click', () => {
        nn.save()
    })

    for (const flight of testData) {
        testFlight(flight)
    }
}


//bereken de accuracy met behulp van alle test data
async function testFlight(flight) {
    // prediction
    let flightData = {
        time: flight.Time,
        lenght: flight.Length,
        dayOfWeek: flight.DayOfWeek,
        airportFrom: airportsFrom.indexOf(flight.AirportFrom),
        airportTo: airportsTo.indexOf(flight.AirportTo),
        airline: airlines.indexOf(flight.Airline)
    }

    nn.classify(flightData, (error, result) => afterPredictionHandler(result, error, flightData))
}

function afterPredictionHandler(result, error, flight) {
    if (result !== undefined) {   
        // vergelijk de result met het echte label
        if (result[0].label == "OnTime") {
            if (flight.Class == 1) {
                //top-predictontime_left-actualdelay
                predictWrongOnTime++
            } else {
                //top-predictontime_left-actualontime
                predictGoodOnTime++
            }
        } else {
            if (flight.Class == 0) {
                //top-predictdelay_left-actualontime
                predictWrongDelayed++
            } else {
                //top-predictdelay_left-actualdelay
                predictGoodDelayed++
            }
        }
    } else if (error !== undefined) {
        console.error(error)
    }

    document.getElementById('accuracy').innerHTML = `Accuracy: ${(predictGoodOnTime + predictGoodDelayed) / testData.length} - Got Right: ${(predictGoodOnTime + predictGoodDelayed)}; Out of total: ${testData.length}`

    document.getElementById('top-predictdelay_left-actualdelay').innerHTML = predictGoodDelayed
    document.getElementById('top-predictdelay_left-actualontime').innerHTML = predictWrongDelayed
    document.getElementById('top-predictontime_left-actualdelay').innerHTML = predictWrongOnTime
    document.getElementById('top-predictontime_left-actualontime').innerHTML = predictGoodOnTime

}

loadData()