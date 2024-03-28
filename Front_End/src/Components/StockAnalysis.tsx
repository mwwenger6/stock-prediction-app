import React, { useState, useEffect } from 'react';
import config from '../config';
import Spinner from "./Spinner";
import ReactECharts from "echarts-for-react";
import DataTable from "react-data-table-component";
import { Tooltip } from 'react-bootstrap';
import InformationWidget from "./InformationWidget";

interface TechnicalData {
    meanPercentReturn: number;
    name: string;
    percentVolatility: number;
    pricePoints: number[];
    ticker: string;
}

const StockAnalysis = () => {
    const useClosePrices = true;
    const ONEMONTH = 30;
    const SIXMONTHS = 182;
    const ONEYEAR = 365;

    const [technicalData, setTechnicalData] = useState<TechnicalData[]>([]);
    const [selectedOption, setSelectedOption] = useState<number>(30);

    const fetchTechnicalData = async (numDaysLookback : number) => {
        try {
            const response = await fetch(config.getTechnicalAnalysis(numDaysLookback, useClosePrices));
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

    useEffect(() => {
        fetchTechnicalData(selectedOption);
    }, []);

    const handleDropdownChange = (event: any) => {
        const selectedValue = event.target.value;
        if((selectedValue == ONEMONTH || selectedValue == SIXMONTHS || selectedValue == ONEYEAR) && selectedValue != selectedOption){
            setSelectedOption(selectedValue);
            setTechnicalData([])
            fetchTechnicalData(selectedValue);
        }
    }

    const tableCustomStyles = {
        rows: {
            style: {
                maxHeight: '130px',
            },
        },
        headCells: {
            style: {
                fontSize: '1.5em',
                fontWeight: 'semi-bold',
                justifyContent: 'center',
                padding:0,
            },
        },
        cells: {
            style: {
                justifyContent: 'center',
                padding:0,
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
            name: 'Closing Price Graph',
            cell: (row: TechnicalData) => <div style={{ height: '200px', width: '100%' }}> <ReactECharts option={createGraph(row.pricePoints)} style={{ height: '100%', width: '100%' }} /></div> ,
            width: '325px'
        },
    ];

    return (
        <div className={"row m-md-2 m-1 border rounded border-gray"}>
            <div className={'d-flex justify-content-between my-3'}>
            <div style={{width: '150px'}} className={'d-flex align-items-center ms-2'}> <InformationWidget tooltipText={'Stats are based on day to day price changes'} iconColor={'info'} iconSize={'medium'}/> </div>
                <h3> Featured Stocks Statistics</h3>
                <div style={{width: '150px'}}>
                    <select className={'form-select form-select-sm mt-1'} value={selectedOption} onChange={handleDropdownChange}>
                        <option value={ONEMONTH}>One Month</option>
                        <option value={SIXMONTHS}>Six Months</option>
                        <option value={ONEYEAR}>One Year</option>
                    </select>   
            </div>
      </div>
            <hr className={'mb-0'}/>
            {technicalData.length > 0 ? (
                <div className={'overflow-auto'} style={{maxHeight: '70vh'}}>
                    <DataTable
                        pagination
                        paginationRowsPerPageOptions={[10, 20]}
                        columns={technicalDataHeaders}
                        data={technicalData}
                        customStyles={tableCustomStyles}
                    />
                </div>
            ) : (
                <div style={{height: '70vh'}}>
                    <Spinner/>
                </div>
            )}
        </div>
    );
};

export default StockAnalysis;