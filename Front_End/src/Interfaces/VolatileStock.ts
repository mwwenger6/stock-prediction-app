interface VolatileStock {
    ticker: string;
    name: string;
    price: number;
    percentChange: number;
    isPositive: boolean;
}
export default VolatileStock;