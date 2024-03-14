import React, { useEffect } from "react";
import { useParams } from 'react-router-dom';
import config from "../config";

const VerifyUser = () => {
  // Extract the `code` parameter from the URL
  let { code } = useParams();
  useEffect(() => {
    const verifyUser = async () => {
      if (!code) return; // Guard clause to ensure code is not undefined or empty

      try {
        const response = await fetch("https://localhost:7212/Home/VerifyUser/" + code, {
          method: "POST",
        });

        if (response.ok) {
          const data = await response.json();
          console.log("Verification response:", data);
          // Additional logic here for handling verification success, if needed
        } else {
          console.error("Verification failed:", response.statusText);
        }
      } catch (error) {
        console.error("Verification error:", error);
      }
    };

    verifyUser();
  }, [code]);
  return (
    <div className="floatingDiv" style={{ textAlign: "center", marginTop: "50px" }}>
      <h1 style={{ color: "green", fontSize: "3em" }}>
        Your Account Has Been Verified
      </h1>
    </div>
  );
};
export default VerifyUser;
