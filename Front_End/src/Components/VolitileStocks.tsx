import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import User from '../Interfaces/User';
import endpoints from '../config';
import VolatileStock from '../Interfaces/VolatileStock';

interface VolitileStocksProps {
    isGainers: boolean;
}

const VolitileStocks: React.FC<VolitileStocksProps> = (props) => {
    const [biggestGainers, setBiggestGainers] = useState<VolatileStock[]>([]); 
    const [biggestLosers, setBiggestLosers] = useState<VolatileStock[]>([]); 
    useEffect(() => {
        const fetchGainers = async () => {
            const response = await fetch(endpoints.getBiggestGainers());
            const data = await response.json();
            setBiggestGainers(data);
        };
        const fetchLosers = async () => {
            const response = await fetch(endpoints.getBiggestLosers());
            const data = await response.json();
            setBiggestLosers(data);
        };
        fetchGainers();
        fetchLosers();
    });

    return (
        <div className="floatingDiv" style={{maxHeight: '500px'}}>
            {props.isGainers ? (
                <>
                    <h3>Today's Biggest Gainers</h3>
                    <hr/>
                    <div className="overflow-auto" style={{ maxHeight: '400px' }}>
                        {biggestGainers.map((stock, index) => (
                            <div key={index} className="floatingDiv m-auto my-2" style={{ minWidth: '200px', width: '200px' }}>
                                <p>{stock.ticker}</p>
                                <p>${stock?.price?.toFixed(2)}</p>
                                <p style={{ color: 'green' }}>{stock.percentChange}%</p>
                            </div>
                        ))}
                    </div>
                </>
            ) : (
                <>
                    <h3>Today's Biggest Losers</h3>
                    <hr/>
                    <div className="overflow-auto" style={{ maxHeight: '400px' }}>
                        {biggestLosers.map((stock, index) => (
                            <div key={index} className="floatingDiv m-auto my-2" style={{ minWidth: '200px', width: '200px' }}>
                                <p>{stock.ticker}</p>
                                <p>${stock?.price?.toFixed(2)}</p>
                                <p style={{ color: 'red' }}>{stock.percentChange}%</p>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
};

export default VolitileStocks;
