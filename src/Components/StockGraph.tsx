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
  const interval3 = '1day'
  const getData = GetTimeSeriesData;
  const [options, setOptions] = useState({});
  const [timeSeriesData, setTimeSeriesData] = useState({});
  //Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
  const [interval, setInterval] = useState('5min');

  useEffect(() => {

    if(symbol === undefined) return 
    
    //Fetch price data on load
    const fetchData = async () => {
        try {
            const timeSeriesData  = await getData(symbol, interval)
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
            if(interval === '1day') intervalLabel = '1 Month'
            else if(interval === '30min') intervalLabel = '1 Day'
            else intervalLabel = '1 Hour'

            console.log(dates)
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
        }
      };

      fetchData();
  }, [interval]);

  //limited to 8 api calls per minute
  return (
    <> 
      <ReactECharts option={options}/>
      <div className='d-flex flex-row justify-content-center'>
          <div className='mx-3'> <Button variant='outline-secondary' onClick={() => interval != interval1 ? setInterval(interval1) : ''}> 1 Hour </Button> </div>
          <div className='mx-3'> <Button variant='outline-secondary' onClick={() => interval != interval2 ? setInterval(interval2) : ''}> 1 Day </Button> </div>
          <div className='mx-3'> <Button variant='outline-secondary' onClick={() => interval != interval3 ? setInterval(interval3): ''}> 1 Month </Button> </div>
          <div className='mx-3'> <CsvDownload className='btn btn-outline-secondary' data={timeSeriesData} filename="stock_data.csv"> Download CSV </CsvDownload> </div>
      </div>
    </>
    )
};

export default StockGraph;
