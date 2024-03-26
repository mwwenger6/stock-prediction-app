import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import PersonalData from '../Data/personalGraphData.json';

interface TimeSeriesData {
  "Meta Data": {},
  "Time Series (5min)": {
    [key: string]: {
      "1. open": string;
      "4. close": string;
    }
  }
}

const StockGraph = () => {
  const [options, setOptions] = useState({});

  useEffect(() => {
    const timeSeries = PersonalData["Time Series (5min)"] as TimeSeriesData["Time Series (5min)"];
    const categories = Object.keys(timeSeries).sort();
    const openValues = categories.map(time => parseFloat(timeSeries[time]["1. open"]));
    const closeValues = categories.map(time => parseFloat(timeSeries[time]["4. close"]));

    let color = 'grey';
    if (closeValues[0] > closeValues[closeValues.length - 1]) {
      color = 'red';
    } else if (closeValues[0] < closeValues[closeValues.length - 1]) {
      color = 'green';
    }

    const allValues = openValues.concat(closeValues);
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);

    const mostCurrentPrice = closeValues[closeValues.length - 1];
    const formattedCurrentPrice = mostCurrentPrice.toLocaleString(); // Format with commas

    const newOptions = {
      title: {
        text: `Current Price: $${formattedCurrentPrice}`,
        textStyle: {
          color: color
        }
      },
      xAxis: {
        type: 'category',
        data: categories
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
  }, []);

  return (
    <div className='floatingDiv'>
        <h3>Performance Graph</h3>
        <hr/>
        <ReactECharts option={options} style={{ height: '400px' }}/>
    </div>
  )
};

export default StockGraph;
