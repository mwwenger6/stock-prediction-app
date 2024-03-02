// Services/ErrorLogService.ts
import ErrorLog from "../Interfaces/ErrorLog";
import config from "../config";

export const getErrorLogs = async (): Promise<ErrorLog[]> => {
  try {
    const response = await fetch(config.getErrorLogs);
    if (!response.ok) {
      throw new Error(`Error fetching error logs: ${response.statusText}`);
    }
    const logsData: ErrorLog[] = await response.json();

    // Assuming the backend returns logs in the correct format, otherwise map/transform here
    return logsData;
  } catch (error) {
    console.error('Error fetching error logs:', error);
    return []; // Return an empty array or handle error appropriately
  }
};
