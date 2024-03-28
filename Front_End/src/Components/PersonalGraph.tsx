import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import User from '../Interfaces/User';
import endpoints from '../config';
import Spinner from "./Spinner";

interface PersonalGraphProps {
    user: User;
}

const PersonalGraph: React.FC<PersonalGraphProps> = (props) => {
    const [options, setOptions] = useState({});
    const [noPersonalStocks, setNoPersonalStocks] = useState<boolean | undefined>(undefined);
    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch(endpoints.getUserStockData(props.user.id));
            const data = await response.json();
            if(data == null) {
                setNoPersonalStocks(true)
                return
            }
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
                    data: categories,
                    axisLabel: {
                        show: false // Hide x-axis labels
                    }
                },
                yAxis: {
                    type: 'value',
                    min: minValue,
                    max: maxValue,
                },
                series: [
                    {
                        data: stockData,
                        type: 'line',
                        color: color,
                        symbol: 'none'
                    }
                ]
            };

            setNoPersonalStocks(false)
            setOptions(newOptions);
        };

        fetchData();
    }, [props.user.id]);

    return (
        <div className='floatingDiv'>
            <h3>Performance Graph</h3>
            <hr />
            {noPersonalStocks === undefined ? (
                <Spinner size={'large'} /> )
                : ( noPersonalStocks ? ( <h5>No personal stocks found</h5>)
                    : ( <ReactECharts option={options} style={{ height: '400px' }} /> )
            )}
        </div>
    );
};

export default PersonalGraph;
