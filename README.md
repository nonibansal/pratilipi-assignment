###Idea is to get new recommendations for users and recommend them pratilipi.


The past data of interactions tells which user read which pratilipi and how much (can be thought of as how much did they like).
The metadata tells us information about pratilipi -> author, category (multi) and year of published, now much features are generated such as how many reads per pratilipi, how many reads more than 50% per pratilipi, unique users read, unique users who read more than 50%, average completion rate.



#Two Approaches Tried :

 - ###Collaborative Filtering :

    Given a user and his/her past read, figure out users like him and suggest him what they read.
    
    ####Drawbacks :
        1. Huge amounts of unique users and pratilips, to counter this I only took users who read alteast 20 different pratilipis and took pratilips which were read by atleast 20 different users.
        
    ####Improvements :
        1. Trying different values than 20, needs computational power for analysis.
        
 - ###Content Based Filtering :
    
     Given a pratilipi try to figure put similar pratilipi from metadata dataset that we created.
    
    

Now as a final model, we will try to take recommendation from both the approcahes giving more weightage to first one and less to second one. Given a user first we check can we generate something from Collaborative filtering if yes we take 10 from them and 10 from second join and show the one he hasn't read till now and sorted according to score.
If his/her recommendations cannot be taken from frist we rely on second and generate recommendations.




Drawbacks of only using Content Based filtring would be we don't have many features describing a pratilipi apart from category, author and published year.




