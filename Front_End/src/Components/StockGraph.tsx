import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import GetTimeSeriesData from "../Services/GetTimeSeriesData";
import CsvDownload from "react-json-to-csv";
import { Button } from 'react-bootstrap';
import Spinner from "./Spinner";
import TimeSeriesData from "../Interfaces/TimeSeriesData";
import endpoints from '../config';
import User from "../Interfaces/User";
import timeSeriesData from "../Interfaces/TimeSeriesData";

interface StockGraphProps {
  symbol: string;
  isFeatured: boolean;
  user: User | null,
}

const StockGraph = (props : StockGraphProps) => {

    const getData = GetTimeSeriesData;
    const intervals =      ['5min',   '30min', '4h',     '1day',    '1month']
    const intervalLabels = ['1 Hour', '1 Day', '1 Week', '1 Month', '1 Year']

    const [options, setOptions] = useState({});
    const [timeSeriesData, setTimeSeriesData] = useState({});
    //Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    const [currInterval, setCurrInterval] = useState(intervals[2]);
    const [showError, setShowError] = useState(false)
    const [marketClosed, setMarketClosed] = useState(false)
    const [graphLoading , setGraphLoading] = useState(true)
    const [ticker, setTicker] = useState('')
    const [percentChange, setPercentChange] = useState('')
    const [color, setColor] = useState('grey')
    const [showPrediction, setShowPrediction] = useState(false)
    const [predictions, setPredictions] = useState([]);

    function stockMarketClosed() {
        // Get the current time in the specified time zone
        const timeZone = 'America/New_York';
        const now = new Date (new Date().toLocaleString('en-US', { timeZone }));

        const dayOfWeek = now.getDay(); // 0 (Sunday) to 6 (Saturday)
        const currentHour = now.getHours();
        const currentMinute = now.getMinutes();

        // Check if it's a weekend (Saturday or Sunday)
        if (dayOfWeek === 0 || dayOfWeek === 6) return true;

        // Check if it's before 9:30 AM or after 4:00 PM ET
        if (currentHour < 9 || (currentHour === 9 && currentMinute < 30) || currentHour >= 16) return true;

        // The stock market is open
        return false;
    }

    function shuffle(array: []) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    function getFormattedDate(datetime: string | Date){
        let year : any = 'numeric';
        let hour : any = undefined;
        let minute : any = undefined;
        if(currInterval == '5min' || currInterval == '30min'){
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
        
            try {
                const response = await fetch(endpoints.predict(props.symbol, 30));
                const jsonData = await response.json();
                setPredictions(jsonData);
                console.log(jsonData);

            }
            catch (error) {
                console.error('Error fetching predictions:', error);
                setShowError(true);
            }
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
                setMarketClosed(stockMarketClosed())

                if (showPrediction) {
                    setCurrInterval(intervals[3]);
                }

                const timeSeriesData  = await getData(props.symbol, currInterval, marketClosed)
                if (timeSeriesData.status == 'error')
                    throw "Unable to get data";

                setShowError(false);

                const timeSeries =  timeSeriesData.values;
                const values = timeSeries.map((item : timeSeriesData) => ({
                    date: getFormattedDate(item.datetime),
                    value: parseFloat(item.open)
                })).reverse();

                // Calculate min and max for Y-axis
                var prices = values.map((item: { value: number, date: any}) => item.value);
                var minValue = Math.floor(Math.min(...prices));
                var maxValue = Math.ceil(Math.max(...prices));

                // If showing a prediction, add the new values to the graph
                // and set the interval to the last month
                const newValues = [];
                if (showPrediction) {
                    let lastDate = new Date(values[values.length-1].date)
                    const newPrices = [187.21963297489884, 188.58384940643663, 187.93998478984298, 189.11436896160262, 188.84965260994042, 187.36609984619832, 185.77344074321073, 186.98540281739267, 184.24540856785242, 188.29042658050895, 191.04728061924155, 181.56758224364995, 192.50320017491174, 191.0724684106699, 192.97252997139736, 193.0672898663342, 193.69445772151175, 192.57864127836467, 190.71306009231097, 188.21145320857093, 182.39096761317484, 184.29850128031308, 186.16351186914994, 185.53661572693278, 186.27703354400185, 185.10228255974567, 185.7324936000796, 180.93893355283825, 182.59532293068588, 184.68495862387908];
                    for (let i = 0; i < newPrices.length; i++) {
                        const date = new Date();
                        date.setDate(lastDate.getDate() + i);
                        const newValue = {
                            date: getFormattedDate(date),
                            value: newPrices[i]
                        };
                        newValues.push(newValue);
                    }
                    // calculate min and max for Y-axis
                    prices = [...prices, ...newPrices];
                    minValue = Math.floor(Math.min(...prices));
                    maxValue = Math.ceil(Math.max(...prices));
                }


                //Calculate ROI/profits
                let lineColor = 'grey'
                let firstPrice = prices[0]
                let lastPrice = prices[prices.length-1]

                if (firstPrice < lastPrice)
                    lineColor = 'green'
                else if (firstPrice > lastPrice)
                    lineColor = 'red'

                let change : string = (((lastPrice/firstPrice) - 1) * 100).toFixed(2)
                if(change.substring(0,1) != '-') change = "+" + change

                setPercentChange(change)
                setTicker(props.symbol)
                setColor(lineColor)

                //if showing a prediction, show different x axis data and show two different lines
                let graphData;
                let xAxisData;
                if(showPrediction){
                    // Add the last price point from values to the beginning of newValues
                    newValues.unshift({
                        date: values[values.length - 1].date,
                        value: values[values.length - 1].value
                    });

                    let placeholders = []
                    for (let i = 0; i < values.length - 1; i++) {
                        placeholders.push('-');
                    }

                    graphData =
                    [{
                        data: values.map((item: { date: Date; value: any; }) => item.value),
                        type: 'line',
                        color: lineColor
                    },
                    {
                        data: [...placeholders, ...newValues.map((item) => item.value)],
                        type: 'line',
                        color: 'blue'
                    }]
                    xAxisData = [
                        ...values.map((item: { date: Date }) => item.date),
                        ...newValues.slice(1, newValues.length).map((item) => item.date)
                    ];
                }
                else{
                    graphData = [{
                        data: values.map((item: { date: Date; value: any; }) => item.value),
                        type: 'line',
                        color: lineColor
                    }]
                    xAxisData = values.map((item: { date: Date; value: any; }) => item.date);
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
                  series: graphData
                };

                setTimeSeriesData(timeSeries)
                setOptions(newOptions);
                setGraphLoading(false)
            }
            catch (error) {
                console.error('Error fetching prices:', error);
                setShowError(true);
            }
          };
      
      fetchData();
    }, [currInterval, props.symbol, showPrediction]);

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
      <p className={marketClosed && (currInterval == intervals[0] || currInterval == intervals[1]) ? "text-danger" : "text-white"}> * Graph prices reflect the last time the stock market was open </p>
        <div className='d-flex row justify-content-center'>
          {intervals.map((interval, i) => (
            <div className='col-auto' key={i}>
                <Button
                    className={`btn ${currInterval === intervals[i] ? 'btn-secondary text-light' : 'btn-outline-secondary'} ${showPrediction ? 'disabled' : ''}`}
                    variant=''
                    onClick={() => {
                        if (!showPrediction && currInterval !== intervals[i]) {
                            setShowPrediction(false);
                            setCurrInterval(intervals[i]);
                        }
                    }}
                    disabled={showPrediction}
                >
                    {intervalLabels[i]}
                </Button>
            </div>
          ))}
          <div className='col-auto'>
              <Button
                  className={`btn btn-outline-secondary`}
                  variant=''
                  onClick={() => {
                      setShowPrediction(!showPrediction)
                  }}>
                  {showPrediction ? "Hide" : "Show"} Prediction
              </Button>
          </div>
      </div>
      <div className='row mt-3 justify-content-center'>
        <div className='col-auto'>
            <CsvDownload
                className={`btn btn-outline-success ${showError ? 'disabled' : ''}`}
                data={timeSeriesData}
                filename="stock_data.csv">
                Download CSV
            </CsvDownload>
        </div>
        <div className='col-auto'>
            <Button className={"btn btn-outline-success"}
                    variant=''>
                Add To Watchlist
            </Button>
        </div>
      </div>
    </>
    )
};

export default StockGraph;
