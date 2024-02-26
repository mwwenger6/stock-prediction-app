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
        let hour : any = undefined;
        let minute : any = undefined;
        if(currInterval == '5min' || currInterval == '30min')
            hour = minute = '2-digit'
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: hour,
            minute: minute
        }).format(new Date(datetime))
    }

    function getPredictions() {

    }

    useEffect(() => {

        if(props.symbol === undefined) return

        //Fetch price data on load
        const fetchData = async () => {
            try {
                setMarketClosed(stockMarketClosed())
                setGraphLoading(true)

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
                const prices = values.map((item: { value: number, date: any}) => item.value);
                const minValue = Math.floor(Math.min(...prices));
                const maxValue = Math.ceil(Math.max(...prices));

                //Generate fake new prices and dates... for now
                const newPrices = [...shuffle(prices.slice(1, prices.length - 1)), prices[prices.length-1]].reverse();
                let lastDate = new Date(values[values.length-1].date)
                const newValues = [];
                for (let i = 0; i < newPrices.length; i++) {
                    const date = new Date();
                    date.setDate(lastDate.getDate() + i);
                    const newValue = {
                        date: getFormattedDate(date),
                        value: newPrices[i]
                    };
                    newValues.push(newValue);
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
                    let placeholders = []
                    for (let i = 0; i < values.length && i < newValues.length; i++) {
                        placeholders.push('-')
                        if (values[i].date === newValues[i].date)
                            break;
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
      
    // tests predictions from our API. Simply logs the json response
    // to the console  
    // const fetchPredictions = async () => {
    //
    //     try {
    //       const response = await fetch(endpoints.predict(props.symbol, 30));
    //       const jsonData = await response.json();
    //       console.log(jsonData);
    //     }
    //     catch (error) {
    //       console.error('Error fetching predictions:', error);
    //       setShowError(true);
    //     }
    //
    //   };
    //   if(props.isFeatured)
    //       fetchPredictions();
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
                      className={`btn ${currInterval === intervals[i] ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                      variant=''
                      onClick={() => {
                          if(currInterval != intervals[i]) {
                              setShowPrediction(false)
                              setCurrInterval(intervals[i])
                          }
                      }}>
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
          <div className='col-auto'>
              <CsvDownload
                  className={`btn btn-outline-secondary ${showError ? 'disabled' : ''}`}
                  data={timeSeriesData}
                  filename="stock_data.csv">
                  Download CSV
              </CsvDownload>
          </div>
      </div>
    </>
    )
};

export default StockGraph;
