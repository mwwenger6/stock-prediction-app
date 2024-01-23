import { Image } from 'react-bootstrap';
import myImage from './thumbnail.webp';
import React, { useState, useEffect } from 'react';
import GetNewsArticles from '../Services/GetNewsArticles';

// Define an interface for the article data
interface Article {
    title: string;
    url: string;
    urlToImage: string;
    description: string;
}

const News = () => {

    const getArticles = GetNewsArticles;
    const [showError, setShowError] = useState(false);
    const [products, setProducts] = useState<Article[]>([]);

    useEffect(() => {
    // Fetch article data on load
    const fetchData = async () => {

        try {
            const data = await getArticles();
            var articles = data.articles as Article[];

            // Update the products state variable with the articles array
            setProducts(articles);

        }
        catch (error) {
            console.error('Error fetching articles:', error);
            setShowError(true);
        }

    }
    fetchData();
    }, []);

    return(
        <>
            <h1>Featured News Artocles</h1>
            <div className="articles-container">
                {products.map((article, index) => (
                    // Render each article as a component
                    <div className="article" key={index}>
                        <Image src={article.urlToImage} alt={article.title} />
                        <h2>{article.title}</h2>
                        <p>{article.description}</p>
                        <a href={article.url}>Read more</a>
                    </div>
                ))}
            </div>
        </>
    );


};

export default News;
