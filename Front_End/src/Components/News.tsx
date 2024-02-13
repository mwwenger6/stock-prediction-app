import { Image } from 'react-bootstrap';
import myImage from './thumbnail.webp';
import React, { useState, useEffect } from 'react';
import GetNewsArticles from '../Services/GetNewsArticles';
import Article from "../Interfaces/Article";


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
                console.log(articles)
                // Update the products state variable with the articles array
                setProducts(articles);

            } catch (error) {
                console.error('Error fetching articles:', error);
                setShowError(true);
            }

        }
        fetchData();
    }, []);

    return (
        <>
            <h1 className="my-3"> Featured News Articles </h1>
            <hr/>
            <div className="row">
                {products.map((article, index) => (
                    <div className="col-lg-6 col-12">
                        <div className="article bg-white m-3 d-flex flex-column justify-content-between" key={index} style={{ height: '600px', maxHeight: '600px' }} >
                            <Image className="m-auto my-2 border border-black" src={article.urlToImage} alt={article.title} style={{ maxWidth: '500px', maxHeight: '350px'}} />
                            <h2 className="title">{article.title}</h2>
                            <p className="description"> {article.description?.length > 150 ? `${article.description.slice(0, 150)}...` : article.description}</p>
                            <div> <a className="readMore" target="_blank" href={article.url}>Read more</a> </div>
                        </div>
                    </div>
                ))}
            </div>
        </>
    );
}
export default News;
