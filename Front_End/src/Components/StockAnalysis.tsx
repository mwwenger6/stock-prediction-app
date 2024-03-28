import React, { useState, useEffect } from 'react';
import config from '../config';
import Spinner from "./Spinner";
import ReactECharts from "echarts-for-react";
import DataTable from "react-data-table-component";

interface TechnicalData {
    meanPercentReturn: number;
    name: string;
    percentVolatility: number;
    pricePoints: number[];
    ticker: string;
}

const StockAnalysis = () => {
    const daysLookBack = 30;
    const useClosePrices = true;
    const [technicalData, setTechnicalData] = useState<TechnicalData[]>([]);
    const [selectedOption, setSelectedOption] = useState<string>('1 Week');

    useEffect(() => {
        const fetchTechnicalData = async () => {
            try {
                const response = await fetch(config.getTechnicalAnalysis(daysLookBack, useClosePrices));
                if (!response.ok) {
                    throw new Error('Failed to fetch technical data');
                }
                const data = await response.json();
                data.forEach((stock: TechnicalData) => {
                    stock.pricePoints.reverse();
                });
                setTechnicalData(data);
            } catch (error) {
                console.error('Error fetching technical data:', error);
            }
        };

        fetchTechnicalData();
    }, []);

    const handleDropdownChange = (event: any) => {
        const selectedValue = event.target.value;
        setSelectedOption(selectedValue);
      };

    const tableCustomStyles = {
        headCells: {
            style: {
                fontSize: '1.5em',
                fontWeight: 'semi-bold',
                justifyContent: 'center',
            },
        },
        cells: {
            style: {
                justifyContent: 'center',
            },
        },
    }

    const createGraph = (pricePoints: number[]) => {
        return {
            xAxis: {
                type: 'category',
                axisLabel: {
                    show: false // Hide x-axis labels
                }
            },
            yAxis: {
                type: 'value',
                min: Math.floor(Math.min(...pricePoints)),
                max: Math.ceil(Math.max(...pricePoints)),
            },
            series: [{
                type: 'line',
                data: pricePoints,
                color: pricePoints[0] < pricePoints[pricePoints.length - 1] ? 'green' : 'red',
                symbol: 'none'
            }],
        };
    };

    const technicalDataHeaders = [
        {
            name: 'Stock',
            cell: (row: TechnicalData) =>
                <h5 className={'text-center'}>
                {row.name} ({row.ticker})
                </h5>,
            sortable: false,
        },
        {
            name: 'Avg Return',
            cell: (row: TechnicalData) => (
                <h5 className={'text-center'}>
                    {row.meanPercentReturn.toString()[0] !== '-' ? '+' + row.meanPercentReturn.toFixed(2) : row.meanPercentReturn.toFixed(2) }%
                </h5>
            ),
            sortable: false,
        },
        {
            name: 'Volatility',
            cell: (row: TechnicalData) =>
                <h5 className={'text-center'}>
                    {row.percentVolatility.toFixed(2)}%
                </h5>,
            sortable: false,
        },
        {
            name: 'Graph',
            cell: (row: TechnicalData) => <div style={{ height: '200px', width: '100%' }}> <ReactECharts option={createGraph(row.pricePoints)} style={{ height: '100%', width: '100%' }} /></div> ,
            width: '350px'
        },
    ];

    return (
        <div className={"row m-md-2 m-1 border rounded border-gray"}>
            <div className={'d-flex justify-content-between my-3'}> 
            <div style={{width: '150px'}}> </div>
                <h3> Technical Indicators For Featured Stocks </h3>
                <div style={{width: '150px'}}>
                    <select value={selectedOption} onChange={handleDropdownChange}>
                        <option value="">Select an option</option>
                        <option value="option1">Option 1</option>
                        <option value="option2">Option 2</option>
                        <option value="option3">Option 3</option>
                    </select>   
            </div>
      </div>
            <hr/>
            {technicalData.length > 0 ? (
                <div className={'overflow-auto'} style={{maxHeight: '70vh'}}>
                    <DataTable
                        pagination
                        columns={technicalDataHeaders}
                        data={technicalData}
                        customStyles={tableCustomStyles}
                    />
                </div>
            ) : (
                <div style={{height: '300px'}}>
                    <Spinner/>
                </div>
            )}
        </div>
    );
};

export default StockAnalysis;