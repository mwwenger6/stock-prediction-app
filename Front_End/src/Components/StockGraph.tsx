import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import GetTimeSeriesData from "../Services/GetTimeSeriesData";
import CsvDownload from "react-json-to-csv";
import { Button } from 'react-bootstrap';

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

  const interval1 = '5min'
  const interval2 = '30min'
  const interval3 = '8h'
  const interval4 = '1day'
  const interval5 = '1month'
  const getData = GetTimeSeriesData;
  const [options, setOptions] = useState({});
  const [timeSeriesData, setTimeSeriesData] = useState({});
  //Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
  const [interval, setInterval] = useState(interval3);
  const [showError, setShowError] = useState(false)
  const [marketClosed, setMarketClosed] = useState(false)
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
            const timeSeriesData  = await getData(symbol, interval, marketClosed)

            if (timeSeriesData.status == 'error')
                throw "Unable to get data";
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

            // Determine color based on the first and last close values
            let color = 'grey';
            if (openValues[0] < openValues[openValues.length - 1]) {
              color = 'green';
            } else if (openValues[0] > openValues[openValues.length - 1]) {
              color = 'red';
            }

            //Determine interval label based on actual interval selected
            let intervalLabel;
            if(interval === interval5) intervalLabel = '1 Year'
            else if(interval === interval4) intervalLabel = '1 Month'
            else if(interval === interval3) intervalLabel = '1 Week'
            else if(interval === interval2) intervalLabel = '1 Day'
            else intervalLabel = '1 Hour'


            const newOptions = {
              title: {
                text: `${symbol} Stock Prices (${intervalLabel})`,
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
        }
        catch (error) {
            console.error('Error fetching prices:', error);
            setShowError(true);
        }
      };

      fetchData();
  }, [interval, symbol]);

  //limited to 8 api calls per minute
  return (
    <>
        {showError ?
            <div style={{height: '300px'}} className='align-items-center d-flex'> <h3 className="m-auto"> Unable to get time series data at this time </h3> </div> :
            <ReactECharts option={options}/> }
      <div className='d-flex flex-row justify-content-center'>
          <div className='mx-3'>
              <Button
              className={`btn ${interval === interval1 ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
              variant=''
              onClick={() => interval != interval1 ? setInterval(interval1) : ''}>
              1 Hour
              </Button>
          </div>
          <div className='mx-3'>
              <Button
                  className={`btn ${interval === interval2 ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                  variant=''
                  onClick={() => interval != interval2 ? setInterval(interval2) : ''}>
                  1 Day
              </Button>
          </div>
          <div className='mx-3'>
              <Button
                  className={`btn ${interval === interval3 ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                  variant=''
                  onClick={() => interval != interval3 ? setInterval(interval3) : ''}>
                  1 Week
              </Button>
          </div>
          <div className='mx-3'>
              <Button
                  className={`btn ${interval === interval4 ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                  variant='outline-secondary'
                  onClick={() => interval != interval4 ? setInterval(interval4): ''}>
                  1 Month
              </Button>
          </div>
          <div className='mx-3'>
              <Button
                  className={`btn ${interval === interval5 ? 'btn-secondary text-light' : 'btn-outline-secondary'}`}
                  variant='outline-secondary'
                  onClick={() => interval != interval5 ? setInterval(interval5): ''}>
                  1 Year
              </Button>
          </div>
          <div className='mx-3'>
              <CsvDownload
                  className={`btn btn-outline-secondary ${showError ? 'disabled' : ''}`}
                  data={timeSeriesData}
                  filename="stock_data.csv">
                  Download CSV
              </CsvDownload>
          </div>
      </div>
        {marketClosed && (interval == interval1 || interval == interval2) ? <p className="text-danger mt-4"> * These prices reflect the last time the stock market was open </p> : <> </>}
    </>
    )
};

export default StockGraph;
