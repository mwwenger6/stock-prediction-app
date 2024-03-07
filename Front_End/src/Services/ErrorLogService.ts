import ErrorLog from "../Interfaces/ErrorLog";
import User from "../Interfaces/User";
import endpoints from "../config";

export const getErrorLogs = async (): Promise<ErrorLog[]> => {
  try {
    const response = await fetch(endpoints.getErrorLogs);
    if (!response.ok) {
      throw new Error(`Error fetching error logs: ${response.statusText}`);
    }
    const logsData: ErrorLog[] = await response.json();
      return logsData;
  } catch (error) {
    console.error('Error fetching error logs:', error);
    return [];
  }
};

export const getUsers = async (): Promise<User[]> => {
  try {
    const response = await fetch(endpoints.getUsers); // Now correctly using the endpoint from config
    if (!response.ok) {
      throw new Error(`Error fetching users: ${response.statusText}`);
    }
    const usersData: User[] = await response.json();
    return usersData;
  } catch (error) {
    console.error('Error fetching users:', error);
    return [];
  }
};
