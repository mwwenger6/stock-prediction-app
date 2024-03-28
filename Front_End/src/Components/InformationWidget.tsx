import React from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';
import {faInfoCircle} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

interface InformationWidgetProps {
    tooltipText: string;
    iconColor:  'primary'| 'secondary' | 'success' | 'danger' | 'warning' | 'info' |'light' | 'dark';
    iconSize: 'small' | 'medium' | 'large';
}

const InformationWidget: React.FC<InformationWidgetProps> = (props) => {
    const tooltip = (
        <Tooltip id="tooltip">
            {props.tooltipText}
        </Tooltip>
    );

    let sizeClass = '';
    switch (props.iconSize) {
        case 'small':
            sizeClass = '20px';
            break;
        case 'medium':
            sizeClass = '25px';
            break;
        case 'large':
            sizeClass = '30px';
            break;
        default:
            sizeClass = '25px';
            break;
    }

    return (
        <OverlayTrigger placement="top" overlay={tooltip}>
            <FontAwesomeIcon icon={faInfoCircle} style={{ height: sizeClass}} className={'text-' + props.iconColor}/>
        </OverlayTrigger>
    );
};

export default InformationWidget;