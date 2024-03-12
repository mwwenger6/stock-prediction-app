import React, {useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import GetTimeSeriesData from "../Services/GetTimeSeriesData";
import CsvDownload from "react-json-to-csv";
import { Button } from 'react-bootstrap';
import Spinner from "./Spinner";
import endpoints from '../config';
import User from "../Interfaces/User";
import timeSeriesData from "../Interfaces/TimeSeriesData";
import config from "../config";


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
    const intervals =      ['1 Hour', '1 Day', '1 Week', '1 Month', '1 Year']

    const [options, setOptions] = useState({});
    const [timeSeriesData, setTimeSeriesData] = useState({});
    //Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    const [currInterval, setCurrInterval] = useState(intervals[2]);
    const [showError, setShowError] = useState(false)
    const [graphLoading , setGraphLoading] = useState(true)
    const [ticker, setTicker] = useState('')
    const [percentChange, setPercentChange] = useState('')
    const [color, setColor] = useState('grey')
    const [showPrediction, setShowPrediction] = useState(false)
    const [predictions, setPredictions] = useState([]);
    const [pendingWatchlistRequest, setPendingWatchlistRequest] = useState(false)

    function getFormattedDate(datetime: string | Date){
        let year : any = 'numeric';
        let hour : any = undefined;
        let minute : any = undefined;
        if(currInterval == '1 Hour' || currInterval == '1 Day'){
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

    useEffect(() => {
        if(props.symbol === undefined) return

        // tests predictions from our API. Simply logs the json response
        // to the console  
        const fetchPredictions = async () => {
        
            // try {
            //     const response = await fetch(endpoints.predict(props.symbol, 30));
            //     const jsonData = await response.json();
            //     setPredictions(jsonData);
            //     console.log(jsonData);
            //
            // }
            // catch (error) {
            //     console.error('Error fetching predictions:', error);
            //     setShowError(true);
            // }
        };


        if(props.isFeatured){
            console.log("Fetching predictions for " + props.symbol);
            fetchPredictions();
        }
    },[])

    useEffect(() => {

        if(props.symbol === undefined) return

        //Fetch price data on load
        const fetchData = async () => {
            try {
                const timeSeriesData  = await getData(props.symbol, currInterval, props.marketClosed);

                if (timeSeriesData.status == 'error')
                    throw "Unable to get data";

                setShowError(false);

                //Get time series values
                const timeSeries =  timeSeriesData.values;

                //Get the prices and dates
                var initPrices = timeSeries.map((item : timeSeriesData) => (parseFloat(item.open))).reverse()
                var initDates = timeSeries.map((item : timeSeriesData) => (getFormattedDate(item.datetime))).reverse()

                var newPrices : number [] = []
                var newDates : Date[] = []
                var placeholders : string[] = []

                // If showing a prediction, add the new values to the graph, and set the interval to the last month
                if (showPrediction) {
                    //Get the predicted prices and update the prices array
                    newPrices = await fetch(endpoints.getPredictions(props.symbol)).then(response => response.json());

                    //Get the next newPrice.length open market days and update dates array
                    const json = await fetch(endpoints.getOpenMarketDays(newPrices.length)).then(response => response.json());
                    newDates = json.map((date: string ) => getFormattedDate(date));

                    //Get placeholders for prediction line
                    for (let i = 0; i < initPrices.length - 1; i++) {
                        placeholders.push('-');
                    }
                }

                var combinedPrices = [...initPrices, ...newPrices];
                var combinedDates = [...initDates, ...newDates];

                // calculate min and max for Y-axis
                var minValue = Math.floor(Math.min(...combinedPrices));
                var maxValue = Math.ceil(Math.max(...combinedPrices));

                //Calculate ROI/profits
                let lineColor = 'grey'
                let firstPrice = combinedPrices[0]
                let lastPrice = combinedPrices[combinedPrices.length-1]

                if (firstPrice < lastPrice)
                    lineColor = 'green'
                else if (firstPrice > lastPrice)
                    lineColor = 'red'

                let change : string = (((lastPrice/firstPrice) - 1) * 100).toFixed(2)
                if(change.substring(0,1) != '-') change = "+" + change

                setPercentChange(change)
                setTicker(props.symbol)
                setColor(lineColor)

                let graphData, xAxisData;
                if(showPrediction){
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
                        color: 'blue',
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
                setGraphLoading(false)
            }
            catch (error) {
                console.error('Error fetching prices:', error);
                setShowError(true);
            }
          };

      fetchData();
    }, [currInterval, props.symbol, showPrediction])

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
                console.log('success')
                props.reloadWatchlist()
            } else {
                console.error('Error sending request:', response.statusText);
            }
            setPendingWatchlistRequest(false)
        }
        makeRequest();
    }

  //limited to 8 api calls per minute
  return (
    <>
        {graphLoading ? (
            <Spinner size={'large'} height={'300px'}/>
        ) : (showError ? (
                <div style={{ height: '300px' }} className='align-items-center d-flex'>
                    <h3 className="m-auto"> Unable to get time series data at this time </h3>
                </div>
            ) : (<div>
                 <span className={"float-start display-6 mb-2"}> {props.symbol}(<span className={color == 'red' ? "text-danger" : (color == 'green'? "text-success" : "text-gray")}>{percentChange}%</span>)</span>
                 <ReactECharts option={options} />
                 </div>)
        )}
        <div className='d-flex row justify-content-center'>
          {intervals.map((interval, i) => (
            <div className='col-auto' key={i}>
                <Button
                    className={`btn ${currInterval !== intervals[i] || showPrediction? 'btn-outline-secondary ' : 'btn-secondary text-light'}`}
                    variant=''
                    onClick={() => {
                        if (currInterval !== intervals[i] || showPrediction) {
                            setShowPrediction(false);
                            setCurrInterval(intervals[i]);
                        }
                    }}
                >
                    {intervals[i]}
                </Button>
            </div>
          ))}
          <div className='col-auto'>
              <Button
                  className={`btn ${showPrediction ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                  variant=''
                  onClick={() => {
                      if(!showPrediction) {
                          setShowPrediction(true)
                          setCurrInterval(intervals[3]);
                      }
                  }}>
                  Show Prediction
              </Button>
          </div>
      </div>
      <div className='row mt-3 justify-content-center mb-2'>
        <div className='col-auto'>
            <CsvDownload
                className={`btn btn-outline-success ${showError ? 'disabled' : ''}`}
                data={timeSeriesData}
                filename="stock_data.csv">
                Download CSV
            </CsvDownload>
        </div>
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
