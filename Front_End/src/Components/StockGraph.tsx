import React, {useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import GetTimeSeriesData from "../Services/GetTimeSeriesData";
import { Button } from 'react-bootstrap';
import Spinner from "./Spinner";
import endpoints from '../config';
import User from "../Interfaces/User";
import timeSeriesData from "../Interfaces/TimeSeriesData";


interface StockGraphProps {
    symbol: string;
    isFeatured: boolean;
    user: User | null;
    isWatchlist: boolean;
    reloadWatchlist: () => Promise<void>;
    marketClosed: boolean;
}

const StockGraph = (props : StockGraphProps) => {

    const getData = GetTimeSeriesData;
    const intervals =      props.marketClosed ? ['1 Day', '1 Week', '1 Month', '1 Year'] : ['1 Hour', '1 Day', '1 Week', '1 Month', '1 Year']
    const initialInterval = intervals[2]

    //State variables for view
    const [options, setOptions] = useState({});
    const [currInterval, setCurrInterval] = useState(intervals[2]);
    const [showError, setShowError] = useState(false)
    const [graphLoading , setGraphLoading] = useState(true)
    const [ticker, setTicker] = useState('')
    const [percentChange, setPercentChange] = useState('')
    const [roiColor, setRoiColor] = useState('text-primary')
    const [showPrediction, setShowPrediction] = useState(false)
    const [pendingWatchlistRequest, setPendingWatchlistRequest] = useState(false)
    const [predictionRange, setPredictionRange] = useState(60)

    //Data retrieved
    const [oneMinTimeSeriesData, setOneMinTimeSeriesData] = useState([])
    const [fiveMinTimeSeriesData, setFiveMinTimeSeriesData] = useState([])
    const [dailyTimeSeriesData, setDailyTimeSeriesData] = useState([])
    const [predictions, setPredictions] = useState([])

    //Fetch the time series data for the stock on symbol change
    useEffect(() => {
        const fetchData = async () => {
            if (props.symbol === undefined) return;
            fetchPrices().then((res) => {
                renderGraph(initialInterval, 0, res);
            });
        };
        fetchData();
    }, [props.symbol]);

    //Add or remove stock from user watchlist on click
    function handleWatchlistClick() {
        const makeRequest = async () => {
            setPendingWatchlistRequest(true)

            var response
            var url = props.isWatchlist ? endpoints.removeUserWatchlistStock(props.user != null ? props.user.id : -1, ticker)
                : endpoints.addUserWatchlistStock(props.user != null ? props.user.id : -1, ticker)

            response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            if (response.status == 200) {
                props.reloadWatchlist()
            } else {
                console.error('Error sending request:', response.statusText);
            }
            setPendingWatchlistRequest(false)
        }
        makeRequest();
    }

    //Format the date for the graph, based on interval
    function getFormattedDate(datetime: string | Date, interval: string){
        let year : any = 'numeric';
        let hour : any = undefined;
        let minute : any = undefined;

        if(interval === '1 Hour' || interval === '1 Day'){
            hour = minute = '2-digit'
            year = undefined
        }
        return new Intl.DateTimeFormat('en-US', {
            year: year,
            month: 'short',
            day: 'numeric',
            hour: hour,
            minute: minute
        }).format(new Date(datetime))
    }


    //Fetch time series data
    const fetchPrices = async() => {
        let oneMinTimeSeriesData = await getData(props.symbol, '1min', props.marketClosed, false);
        let fiveMinTimeSeriesData = await getData(props.symbol, '5min', props.marketClosed, props.isFeatured);
        let dailyTimeSeriesData = await getData(props.symbol, '1day', props.marketClosed, props.isFeatured);

        if(props.isFeatured){
            if (oneMinTimeSeriesData === undefined || oneMinTimeSeriesData.status === 'error')
                oneMinTimeSeriesData = [];
            else if(fiveMinTimeSeriesData.length === 0 || dailyTimeSeriesData.length === 0){
                setShowError(true)
                return;
            }
            setShowError(false);
            setPredictions(await fetch(endpoints.getPredictions(props.symbol)).then(response => response.json()))
        }
        else{
            if (oneMinTimeSeriesData === undefined || oneMinTimeSeriesData.status === 'error' ||
                fiveMinTimeSeriesData === undefined || fiveMinTimeSeriesData.status === 'error' ||
                dailyTimeSeriesData === undefined || dailyTimeSeriesData.status === 'error'){
                setShowError(true)
                return;
            }
            setShowError(false);
        }

        setOneMinTimeSeriesData(oneMinTimeSeriesData.reverse());
        setFiveMinTimeSeriesData(fiveMinTimeSeriesData.reverse());
        setDailyTimeSeriesData(dailyTimeSeriesData.reverse());
        return fiveMinTimeSeriesData;
    };

    //Re-render the graph based on the interval and prediction range. init data is passed on initial load to avoid race condition
    const renderGraph = async(interval: string, predictionRange : number, initData : timeSeriesData[] = []) => {
        let showingPrediction = predictionRange > 0;

        //First, get correct amount of data based on interval
        var data : timeSeriesData[] = []

        if(initData.length > 0)
            data = initData

        else if(interval === '1 Hour')
            data = oneMinTimeSeriesData;

        else if(interval === '1 Day')
            data = fiveMinTimeSeriesData.slice(fiveMinTimeSeriesData.length - 78, fiveMinTimeSeriesData.length);

        else if(interval === '1 Week')
            data = fiveMinTimeSeriesData;

        else if(interval === '1 Month')
            data = dailyTimeSeriesData.slice(dailyTimeSeriesData.length - 21, dailyTimeSeriesData.length);

        else if(interval === '1 Year')
            data = dailyTimeSeriesData;

        //Set the prices and dates
        var initPrices = data.map((item : timeSeriesData) => (parseFloat(item.open)));
        var initDates = data.map((item : timeSeriesData) => (getFormattedDate(item.datetime, interval)));

        //Set predicted prices and new dates if showing a prediction
        var newPrices : number [] = []
        var newDates : Date[] = []
        var placeholders : string[] = []

        if (showingPrediction) {
            //Get placeholders for prediction line
            for (let i = 0; i < initPrices.length - 1; i++) {
                placeholders.push('-');
            }
            // store only the amount determined by prediction range
            newPrices = predictions.slice(0, predictionRange);
            const json = await fetch(endpoints.getOpenMarketDays(predictionRange)).then(response => response.json());
            newDates = json.map((date: string ) => getFormattedDate(date, interval));
        }

        //Get the combined prices and dates and calculate graph values/ROI for them
        var combinedPrices = [...initPrices, ...newPrices];
        var combinedDates = [...initDates, ...newDates];

        var minValue : number = Math.min(...combinedPrices)
        var maxValue : number = Math.max(...combinedPrices)

        minValue = parseFloat(minValue.toFixed(2))
        maxValue =  parseFloat(maxValue.toFixed(2))

        //Percent change is predicted end value/last real value if prediction is shown,
        let index : number = showingPrediction ?  initPrices.length-1 : 0;
        let firstPrice = combinedPrices[index]
        let lastPrice = combinedPrices[combinedPrices.length-1]

        let lineColor = firstPrice <= lastPrice ? 'green' : 'red';
        let change : string = (((lastPrice/firstPrice) - 1) * 100).toFixed(2)
        if(change.substring(0,1) != '-') change = "+" + change

        setPercentChange(change)
        setTicker(props.symbol)

        if(showingPrediction){
            setRoiColor('text-primary')
        }
        else{
            lineColor === 'green' ? setRoiColor('text-success') : setRoiColor('text-danger')
        }

        let graphData, xAxisData;
        if(predictionRange > 0){
            //prepend the last price to the new prices
            newPrices = [initPrices[initPrices.length - 1] , ...newPrices]
            graphData =
                [{
                    data: initPrices,
                    type: 'line',
                    color: lineColor,
                    symbol: 'none'
                },
                {
                    data: [...placeholders, ...newPrices],
                    type: 'line',
                    color: '#0d6efd',
                    symbol: 'none'
                }]
            xAxisData = combinedDates;
        }
        else{
            graphData = [{
                data: initPrices,
                type: 'line',
                color: lineColor,
                symbol: 'none'
            },
                {
                    data: []
                }]
            xAxisData = initDates;
        }
        const newOptions = {
            xAxis: {
                type: 'category',
                data: xAxisData
            },
            yAxis: {
                type: 'value',
                min: minValue,
                max: maxValue
            },
            series: graphData,
        };

        setOptions(newOptions);
        setGraphLoading(false);
    }


    return (
        <>
            {graphLoading ? (
                <Spinner size={'large'} height={'300px'}/>
            ) : (showError ? (
                    <div style={{ height: '500px' }} className='align-items-center d-flex'>
                        <h3 className="m-auto"> Unable to get time series data at this time </h3>
                    </div>
                ) : (<div>
                    <span className={"float-start display-6 mb-2"}> {props.symbol}(<span className={roiColor}>{percentChange}%</span>)</span>
                    <ReactECharts option={options} style={{ height: '500px' }} />
                </div>)
            )}
            <div className='d-flex row justify-content-center' >
                {intervals.map((interval, i) => (
                    <div className='col-auto' key={i}>
                        <Button
                            className={`btn ${currInterval !== intervals[i] || showPrediction? 'btn-outline-secondary ' : 'btn-secondary text-light'}`}
                            variant=''
                            onClick={() => {
                                if (currInterval !== intervals[i] || showPrediction) {
                                    setShowPrediction(false);
                                    setCurrInterval(intervals[i]);
                                    renderGraph(intervals[i], 0);
                                }
                            }}
                        >
                            {intervals[i]}
                        </Button>
                    </div>
                ))}

                {props.isFeatured &&
                    <>
                        <div className='col-auto'>
                            <Button
                                className={`btn ${showPrediction ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                                variant=''
                                onClick={() => {
                                    setShowPrediction(true)
                                    renderGraph(intervals[3], predictionRange);
                                }}>
                                Show Prediction
                            </Button>
                        </div>
                        <div className='col-auto'>
                            <input
                                type="range"
                                min={1}
                                max={60}
                                step={1}
                                value={predictionRange}
                                onChange={(event) => {
                                    setShowPrediction(false)
                                    setCurrInterval('')
                                    setPredictionRange(Number(event.target.value))
                                }}
                            />
                            <div className="slider-value">Prediction Range: {predictionRange}</div>
                        </div>
                    </>
                }
            </div>

            <div className='row mt-3 justify-content-center mb-2'>
                {props.user != null && props.isFeatured &&
                    <div className='col-auto'>
                        <Button className={`btn ${pendingWatchlistRequest ? "disabled" : ""} ${props.isWatchlist ? "btn-outline-danger" : "btn-outline-success"}`}
                                variant=''
                                onClick ={() => handleWatchlistClick()}>
                            {props.isWatchlist ? "Remove From Watchlist" : "Add To Watchlist" }
                        </Button>
                    </div>
                }
            </div>
        </>
    )
};

export default StockGraph;