DROP DATABASE IF EXISTS cancertry;
CREATE DATABASE cancertry;
USE cancertry;

CREATE TABLE userstry (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(225),
    email VARCHAR(225),
    password VARCHAR(225)
);
