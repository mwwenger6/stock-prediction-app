import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import GetTimeSeriesData from "../Services/GetTimeSeriesData";
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

  const getData = GetTimeSeriesData;
  const [options, setOptions] = useState({});
  
  //Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month'
  const [interval, setInterval] = useState('1day');

  useEffect(() => {
    if(symbol === undefined) return 
    
    //Fetch price data on load
    const fetchData = async () => {
        try {
            const timeSeriesData  = await getData(symbol, interval)
            const timeSeries =  timeSeriesData.values;
            const openValues = timeSeries.map((item: TimeSeriesData) => parseFloat(item.open));
            const closeValues = timeSeries.map((item: TimeSeriesData) => parseFloat(item.close));

            // Determine color based on the first and last close values
            let color = 'grey';
            if (closeValues[0] > closeValues[closeValues.length - 1]) {
              color = 'red';
            } else if (closeValues[0] < closeValues[closeValues.length - 1]) {
              color = 'green';
            }

            // Calculate min and max for Y-axis
            const allValues = openValues.concat(closeValues);
            const minValue = Math.min(...allValues);
            const maxValue = Math.max(...allValues);

            const newOptions = {
              title: {
                text: `${symbol} Stock Prices (${interval})`,
              },
              xAxis: {
                type: 'category',
                data: '' //Need to show appropriate labels
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
      <div className='d-flex flex-row'> 
        <div className='mx-3'> <h5> Intervals: </h5> </div>
        <div className='justify-content-left mx-3'> <Button variant='outline-secondary' onClick={() => interval != '1min' ? setInterval('1min') : ''}> 1 Minute </Button> </div>
        <div className='justify-content-left mx-3'> <Button variant='outline-secondary' onClick={() => interval != '1h' ? setInterval('1h') : ''}> 1 Hour </Button> </div>
        <div className='justify-content-left mx-3'> <Button variant='outline-secondary' onClick={() => interval != '1day' ? setInterval('1day') : ''}> 1 Day </Button> </div>
        <div className='justify-content-left mx-3'> <Button variant='outline-secondary' onClick={() => interval != '1month' ? setInterval('1month'): ''}> 1 Month </Button> </div>
      </div>
    </>
    )
};

export default StockGraph;
