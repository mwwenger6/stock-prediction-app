import React from "react";

interface SpinnerProps {
    size?: 'small' | 'medium' | 'large';
    height?: string;
}
const Spinner: React.FC<SpinnerProps> = ({ size = 'medium' , height = '100%'}: SpinnerProps) => {
    const getSizeClass = () => {
        switch (size) {
            case 'small':
                return 'spinner-border-sm';
            case 'large':
                return 'spinner-border-lg';
            default:
                return ''; // medium or default size
        }
    };

    return (
        <div className="d-flex justify-content-center align-items-center" style={{ height: height }}>
            <div className={`spinner-border ${getSizeClass()}`} role="status">
                <span className="sr-only"></span>
            </div>
        </div>
    );
}
export default Spinner;