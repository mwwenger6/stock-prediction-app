import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import TempData from '../Data/tempStockData.json';

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
    const timeSeries = TempData["Time Series (5min)"] as TimeSeriesData["Time Series (5min)"];
    const categories = Object.keys(timeSeries).sort();
    const openValues = categories.map(time => parseFloat(timeSeries[time]["1. open"]));
    const closeValues = categories.map(time => parseFloat(timeSeries[time]["4. close"]));

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
        text: 'MSFT Stock Prices',
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

  return <ReactECharts option={options} />;
};

export default StockGraph;
