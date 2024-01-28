import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import GetTimeSeriesData from "../Services/GetTimeSeriesData";
import CsvDownload from "react-json-to-csv";
import { Button } from 'react-bootstrap';
import Spinner from "./Spinner";

interface TimeSeriesData {
      datetime: string;
      high: string;
      open: string;
      close: string;
      volume: string;
}

interface StockGraphProps {
  symbol: string | undefined;
}

const StockGraph = ({ symbol } : StockGraphProps) => {

    const getData = GetTimeSeriesData;
    const intervals =      ['5min',   '30min', '4h',     '1day',    '1month']
    const intervalLabels = ['1 Hour', '1 Day', '1 Week', '1 Month', '1 Year']

    const [options, setOptions] = useState({});
    const [timeSeriesData, setTimeSeriesData] = useState({});
    //Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    const [currInterval, setCurrInterval] = useState(intervals[2]);
    const [currIntervalLabel, setCurrIntervalLabel] = useState(intervalLabels[2]);
    const [showError, setShowError] = useState(false)
    const [marketClosed, setMarketClosed] = useState(false)
    const [graphLoading , setGraphLoading] = useState(true)

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


    useEffect(() => {

    if(symbol === undefined) return

    //Fetch price data on load
    const fetchData = async () => {
        try {
            setMarketClosed(stockMarketClosed())
            const timeSeriesData  = await getData(symbol, currInterval, marketClosed)
            if (timeSeriesData.status == 'error')
                throw "Unable to get data";
            console.log(timeSeriesData)
            setShowError(false);
            const timeSeries =  timeSeriesData.values;
            const openValues = (timeSeries.map((item: TimeSeriesData) => parseFloat(item.open))).reverse();
            const dates = (timeSeries.map((item: TimeSeriesData) => {
                let date : Date = new Date(item.datetime)

                let dateEndIndex = 5;
                if(date.toLocaleDateString().at(4) === '/') dateEndIndex=4

                let timeEndIndex = 5;
                if(date.toLocaleTimeString().at(4) === ':') timeEndIndex=4

                return date.toLocaleDateString().substring(0,dateEndIndex) + ' ' + date.toLocaleTimeString().substring(0,timeEndIndex);
            })).reverse();

            // Calculate min and max for Y-axis
            const minValue = Math.floor(Math.min(...openValues));
            const maxValue = Math.ceil(Math.max(...openValues));

            let color : string = 'grey'
            if (openValues[0] < openValues[openValues.length - 1])
              color = 'green'
            else if (openValues[0] > openValues[openValues.length - 1])
              color = 'red'

            let change : string = (((timeSeries[0].open/timeSeries[timeSeries.length-1].open) * 100) - 100).toFixed(2)


            const newOptions = {
              title: {
                text: `${symbol} (${change}%)`,
              },
              xAxis: {
                type: 'category',
                  data: dates
              },
              yAxis: {
                type: 'value',
                min: minValue,
                max: maxValue
              },
              series: [
                {
                  data: openValues,
                  type: 'line',
                  color: color,
                }
              ]
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
    }, [currInterval, symbol]);

  //limited to 8 api calls per minute
  return (
    <>
        {graphLoading ? (
            <Spinner size={'large'} height={'300px'}/>
        ) : (showError ? (
                <div style={{ height: '300px' }} className='align-items-center d-flex'>
                    <h3 className="m-auto"> Unable to get time series data at this time </h3>
                </div>
            ) : (<ReactECharts option={options} /> )
        )}
      <div className='d-flex row justify-content-center'>
          {intervals.map((interval, i) => (
              <div className='col-auto' key={i}>
                  <Button
                      className={`btn ${currInterval === intervals[i] ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                      variant=''
                      onClick={() => {
                          if(currInterval != intervals[i]) {
                              setCurrInterval(intervals[i])
                              setCurrIntervalLabel(intervalLabels[i])
                          }
                      }}>
                      {intervalLabels[i]}
                  </Button>
              </div>
          ))}
          <div className='col-auto'>
              <CsvDownload
                  className={`btn btn-outline-secondary ${showError ? 'disabled' : ''}`}
                  data={timeSeriesData}
                  filename="stock_data.csv">
                  Download CSV
              </CsvDownload>
          </div>
      </div>
      <p className={marketClosed && (currInterval == intervals[0] || currInterval == intervals[1]) ? "text-danger mt-4" : "text-white mt-4"}> * These prices reflect the last time the stock market was open </p>
    </>
    )
};

export default StockGraph;
