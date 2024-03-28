import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import User from '../Interfaces/User';
import endpoints from '../config';

interface PersonalGraphProps {
    user: User;
}

const PersonalGraph: React.FC<PersonalGraphProps> = (props) => {
    const [options, setOptions] = useState({});
    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch(endpoints.getUserStockData(props.user.id));
            const data = await response.json();
            const stockData = data.reverse();
            const categories = stockData.map((_: any, index: number) => `Point ${index + 1}`);
            
            // Decide color based on first and last data points
            let color = 'grey';
            if (stockData[0] < stockData[stockData.length - 1]) {
                color = 'green';
            } else if (stockData[0] > stockData[stockData.length - 1]) {
                color = 'red';
            }
            
            const minValue = Math.min(...stockData);
            const maxValue = Math.max(...stockData);
            const mostCurrentPrice = stockData[stockData.length - 1];
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
                        data: stockData,
                        type: 'line',
                        color: color,
                    }
                ]
            };
            
            setOptions(newOptions);
        };

        fetchData();
    }, [props.user.id]);

    return (
        <div className='floatingDiv'>
            <h3>Performance Graph</h3>
            <hr />
            <ReactECharts option={options} style={{ height: '400px' }} />
        </div>
    );
};

export default PersonalGraph;
