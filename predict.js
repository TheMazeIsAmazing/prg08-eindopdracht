let maxTime = 0
let maxLength = 0

let airlines = [
    "DL",
    "OO",
    "B6",
    "US",
    "FL",
    "WN",
    "CO",
    "AA",
    "YV",
    "EV",
    "XE",
    "9E",
    "OH",
    "UA",
    "MQ",
    "AS",
    "F9",
    "HA"
]

let airportsFrom = [
    "ATL",
    "COS",
    "BOS",
    "OGG",
    "BMI",
    "MSY",
    "EWR",
    "DFW",
    "BWI",
    "CRW",
    "LGB",
    "BIS",
    "CLT",
    "IAH",
    "LAX",
    "JAX",
    "SAV",
    "CLE",
    "FLL",
    "SAN",
    "BHM",
    "ROC",
    "DTW",
    "STT",
    "AUS",
    "DCA",
    "PHX",
    "EYW",
    "IND",
    "JFK",
    "ORD",
    "PBI",
    "SFO",
    "MIA",
    "DSM",
    "SLC",
    "PHL",
    "BZN",
    "GRB",
    "MBS",
    "SBA",
    "TYS",
    "MSP",
    "DEN",
    "SAT",
    "BUF",
    "RIC",
    "SEA",
    "PDX",
    "LAS",
    "IAD",
    "HNL",
    "BDL",
    "MOT",
    "PSE",
    "CPR",
    "SNA",
    "STL",
    "CVG",
    "PIT",
    "HSV",
    "SGF",
    "RDU",
    "MEM",
    "KOA",
    "ELP",
    "SJU",
    "JAN",
    "AEX",
    "LGA",
    "RSW",
    "MDT",
    "GUC",
    "MKE",
    "CAE",
    "GRR",
    "FAR",
    "LIT",
    "OMA",
    "BNA",
    "EVV",
    "RDD",
    "OKC",
    "ITO",
    "SJC",
    "MCO",
    "LBB",
    "CSG",
    "OAK",
    "PHF",
    "ABQ",
    "SMF",
    "FAY",
    "ABI",
    "MSO",
    "MFE",
    "GEG",
    "MSN",
    "TPA",
    "DAY",
    "RNO",
    "PVD",
    "ALB",
    "CHO",
    "ONT",
    "LIH",
    "PSP",
    "LAN",
    "LEX",
    "XNA",
    "GJT",
    "CMH",
    "GSO",
    "PSC",
    "SYR",
    "AVL",
    "MHT",
    "GRK",
    "MCI",
    "TXK",
    "LRD",
    "ABE",
    "LWB",
    "ERI",
    "DAL",
    "ANC",
    "TUS",
    "ROA",
    "MOD",
    "JNU",
    "SBP",
    "CDV",
    "TUL",
    "FSD",
    "FNT",
    "BTV",
    "FCA",
    "GNV",
    "RAP",
    "MDW",
    "FWA",
    "BUR",
    "PNS",
    "RST",
    "HOU",
    "BOI",
    "CRP",
    "BRO",
    "ATW",
    "SHV",
    "SMX",
    "RDM",
    "ORF",
    "GPT",
    "KTN",
    "ICT",
    "SAF",
    "CAK",
    "IDA",
    "MQT",
    "VPS",
    "CHS",
    "MAF",
    "HPN",
    "AVP",
    "AZO",
    "TRI",
    "GSP",
    "HDN",
    "MLU",
    "EUG",
    "AMA",
    "MHK",
    "ISP",
    "CID",
    "MOB",
    "BGR",
    "SRQ",
    "MLI",
    "EKO",
    "LFT",
    "TOL",
    "ECP",
    "PSG",
    "SBN",
    "FAT",
    "ELM",
    "YUM",
    "CLD",
    "FAI",
    "ASE",
    "BTR",
    "BQK",
    "COU",
    "MRY",
    "CEC",
    "CWA",
    "PWM",
    "FLG",
    "TLH",
    "SDF",
    "BFL",
    "CHA",
    "ACV",
    "MGM",
    "ROW",
    "GTR",
    "EWN",
    "ILM",
    "OTZ",
    "SGU",
    "OTH",
    "CMX",
    "SWF",
    "BET",
    "GTF",
    "CMI",
    "MFR",
    "JAC",
    "DLH",
    "ABY",
    "MTJ",
    "SCC",
    "DRO",
    "TEX",
    "FSM",
    "COD",
    "GGG",
    "DBQ",
    "GFK",
    "BKG",
    "AGS",
    "BTM",
    "DHN",
    "TYR",
    "EGE",
    "PIH",
    "VLD",
    "MEI",
    "SIT",
    "MLB",
    "PAH",
    "YAK",
    "DAB",
    "HLN",
    "PIA",
    "SPI",
    "GCC",
    "IPL",
    "TVC",
    "OAJ",
    "EAU",
    "BGM",
    "MYR",
    "HRL",
    "MKG",
    "SUN",
    "LSE",
    "CIC",
    "OME",
    "ITH",
    "LNK",
    "BIL",
    "CYS",
    "LCH",
    "BQN",
    "WRG",
    "BRW",
    "SPS",
    "RKS",
    "TWF",
    "LMT",
    "ACT",
    "PLN",
    "ACY",
    "ADK",
    "SJT",
    "IYK",
    "LWS",
    "BLI",
    "SCE",
    "MMH",
    "LYH",
    "GUM",
    "CDC",
    "ADQ",
    "HTS",
    "PIE",
    "STX",
    "FLO",
    "UTM",
    "CLL",
    "ABR"
]

let airportsTo = [
    "HOU",
    "ORD",
    "CLT",
    "PHX",
    "ATL",
    "BHM",
    "DFW",
    "MEM",
    "GRR",
    "PBI",
    "MCO",
    "SFO",
    "DEN",
    "YUM",
    "BWI",
    "HPN",
    "EWR",
    "JFK",
    "MKE",
    "OAK",
    "IAH",
    "CLE",
    "SYR",
    "SJU",
    "BDL",
    "SAN",
    "DTW",
    "PSP",
    "DCA",
    "LGA",
    "STL",
    "FAY",
    "MSP",
    "BUF",
    "LAS",
    "SGU",
    "SLC",
    "GJT",
    "LAX",
    "VPS",
    "FAR",
    "RKS",
    "BOS",
    "ANC",
    "SNA",
    "ONT",
    "RNO",
    "JAX",
    "GSP",
    "CVG",
    "TPA",
    "SEA",
    "LEX",
    "SMF",
    "CAE",
    "STT",
    "DAY",
    "MDW",
    "RSW",
    "ITO",
    "IAD",
    "ICT",
    "HNL",
    "MIA",
    "CRW",
    "RDU",
    "MHT",
    "FAT",
    "CAK",
    "COS",
    "DAL",
    "TYS",
    "PHL",
    "ABI",
    "MOB",
    "SDF",
    "SAV",
    "MDT",
    "LIT",
    "TUL",
    "ACV",
    "BNA",
    "MCI",
    "MSY",
    "FLL",
    "PVD",
    "OKC",
    "ECP",
    "PHF",
    "AUS",
    "RIC",
    "LIH",
    "ABQ",
    "JAN",
    "PIT",
    "BMI",
    "BTV",
    "RAP",
    "MRY",
    "CSG",
    "SHV",
    "FAI",
    "SJC",
    "PIA",
    "SBN",
    "IND",
    "SGF",
    "ACT",
    "SRQ",
    "ROC",
    "CHO",
    "JAC",
    "SAT",
    "FWA",
    "OMA",
    "PDX",
    "CMH",
    "PWM",
    "CID",
    "TRI",
    "ORF",
    "GTF",
    "TUS",
    "MHK",
    "BUR",
    "MLU",
    "CEC",
    "TEX",
    "MBS",
    "DSM",
    "HRL",
    "LFT",
    "ELP",
    "AEX",
    "CPR",
    "LBB",
    "MYR",
    "ALB",
    "COU",
    "LSE",
    "CHA",
    "MLI",
    "GEG",
    "AZO",
    "MFR",
    "BTR",
    "FLG",
    "KTN",
    "PSC",
    "GSO",
    "OGG",
    "MSN",
    "GPT",
    "PNS",
    "RDM",
    "BZN",
    "DLH",
    "CRP",
    "TXK",
    "KOA",
    "MQT",
    "MAF",
    "TLH",
    "XNA",
    "CWA",
    "SBP",
    "BFL",
    "DRO",
    "WRG",
    "DHN",
    "SPS",
    "AMA",
    "EGE",
    "BET",
    "FCA",
    "EUG",
    "EVV",
    "AVL",
    "HSV",
    "PIE",
    "MLB",
    "SWF",
    "ASE",
    "BGM",
    "MSO",
    "ADK",
    "GRK",
    "SUN",
    "SBA",
    "LGB",
    "CHS",
    "GNV",
    "MOT",
    "LAN",
    "LNK",
    "OME",
    "OTH",
    "ISP",
    "FNT",
    "EAU",
    "ILM",
    "BRW",
    "LCH",
    "IYK",
    "MKG",
    "HDN",
    "BRO",
    "GRB",
    "FSD",
    "LRD",
    "RDD",
    "SPI",
    "ROA",
    "IPL",
    "EYW",
    "SAF",
    "LWS",
    "AGS",
    "CMX",
    "ATW",
    "MGM",
    "GGG",
    "BOI",
    "FLO",
    "COD",
    "ACY",
    "CMI",
    "JNU",
    "AVP",
    "ERI",
    "TYR",
    "DAB",
    "TVC",
    "FSM",
    "IDA",
    "MFE",
    "EKO",
    "ABE",
    "PAH",
    "LMT",
    "YAK",
    "HLN",
    "MMH",
    "ITH",
    "LYH",
    "BIL",
    "EWN",
    "SMX",
    "MEI",
    "OAJ",
    "SCE",
    "CLD",
    "BIS",
    "GFK",
    "MTJ",
    "BQN",
    "BQK",
    "GTR",
    "CDV",
    "BKG",
    "PIH",
    "ROW",
    "PLN",
    "TWF",
    "ELM",
    "GCC",
    "CYS",
    "CDC",
    "ABY",
    "VLD",
    "MOD",
    "STX",
    "OTZ",
    "HTS",
    "BTM",
    "PSE",
    "SCC",
    "RST",
    "DBQ",
    "ADQ",
    "PSG",
    "SIT",
    "GUC",
    "LWB",
    "BGR",
    "CIC",
    "GUM",
    "UTM",
    "CLL",
    "TOL",
    "SJT",
    "BLI",
    "ABR"
]


let fieldTime = document.getElementById('time')
let fieldLength = document.getElementById('length')
let fieldAirline = document.getElementById('airline')
let fieldAirportFrom = document.getElementById('airportFrom')
let fieldAirportTo = document.getElementById('airportTo')
let fieldDay = document.getElementById('day')


function addOptions(data, element) {
    for (let item of data) {
        let option = document.createElement("option");
        option.innerHTML = item;
        element.appendChild(option);
    }
}

addOptions(airlines, fieldAirline);
addOptions(airportsFrom, fieldAirportFrom);
addOptions(airportsTo, fieldAirportTo);

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
nn.load('./model/model.json', modelLoaded)

function modelLoaded() {
    Papa.parse("./data/airlines_delay.csv", {
        // Papa.parse("./data/airlines_delay_2000.csv", {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: results => processData(results.data)
    })
}

function processData(data) {

    for (let flight of data) {

        if (flight.Time > maxTime) {
            maxTime = flight.Time
        }

        if (flight.Length > maxLength) {
            maxLength = flight.Length
        }
    }

    fieldTime.max = maxTime
    fieldTime.value = (Math.floor(maxTime / 2))
    document.getElementById('timeLabel').innerHTML = `(${Math.floor(maxTime / 2)})`

    fieldLength.max = maxLength
    fieldLength.value = (Math.floor(maxLength / 2))
    document.getElementById('lengthLabel').innerHTML = `(${Math.floor(maxLength / 2)})`
}

async function makePrediction(e) {
    e.preventDefault()

    let valueTime = parseInt(fieldTime.value)
    let valueLength = parseInt(fieldLength.value)
    let valueDay = parseInt(fieldDay.value)

    let flightData = {
        time: valueTime,
        lenght: valueLength,
        dayOfWeek: valueDay,
        airportFrom: airportsFrom.indexOf(fieldAirportFrom.value),
        airportTo: airportsTo.indexOf(fieldAirportTo.valueo),
        airline: airlines.indexOf(fieldAirline.value)
    }

    nn.classify(flightData, (error, result) => afterPredictionHandler(result, error))
}
function afterPredictionHandler(result, error) {
    if (result !== undefined) {   
        // vergelijk de result met het echte label
        if (result[0].label == "OnTime") {
            document.getElementById('prediction').innerHTML = `Gefeliciteerd! Je vlucht zal op tijd zijn  :-)`
        } else {
            document.getElementById('prediction').innerHTML = `Helaas! Je vlucht zal vertaging hebben  :'(`
        }
    } else if (error !== undefined) {
        console.error(error)
    }
}

document.getElementById('time').addEventListener('change', (e) => {
    document.getElementById('timeLabel').innerHTML = `(${e.target.value})`
})

document.getElementById('length').addEventListener('change', (e) => {
    document.getElementById('lengthLabel').innerHTML = `(${e.target.value})`
})

document.getElementById('day').addEventListener('change', (e) => {
    document.getElementById('dayLabel').innerHTML = `(${e.target.value})`
})

document.getElementById('form').addEventListener('submit', makePrediction)