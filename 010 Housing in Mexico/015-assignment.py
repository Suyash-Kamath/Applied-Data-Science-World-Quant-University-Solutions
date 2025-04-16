import wqet_grader

wqet_grader.init("Project 1 Assessment")


# Before you start

# Import Matplotlib, pandas, and plotly
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


# Prepare Data : Import

# 1.5.1

df1 =  pd.read_csv('data/brasil-real-estate-1.csv')
df1.head()


wqet_grader.grade("Project 1 Assessment", "Task 1.5.1", df1)

# Inspect the DataFrame using info() and head()
df1.info()
df1.head()


# 1.5.2

# Drop rows with NaN values
df1 = df1.dropna()

# Display the first few rows to confirm the operation
df1.head()


# 1.5.3

df1[["lat", "lon"]] = df1["lat-lon"].str.split(",", expand=True)
df1.info()
df1.head()


# In[24]:


df1["lat"] = df1.lat.astype(float)
df1['lon'] = df1.lon.astype(float)
df1.head



# 1.5.4

df1.loc[:,'state'] = df1.loc[:,'place_with_parent_names'].str.split('|', expand=True)[2]

df1.head()


# 1.5.5

df1["price_usd"]=df1.price_usd.str.replace('$','')
df1["price_usd"]=df1.price_usd.str.replace(',','')



df1['price_usd'] = df1.price_usd.astype(float)
df1.head()



# 1.5.6

df1=df1.drop('lat-lon', axis='columns')
df1=df1.drop('place_with_parent_names', axis='columns')
df1.head()



# 1.5.7

df2 = pd.read_csv('data/brasil-real-estate-2.csv')

df2.info()


# In[36]:


df2.head()


# 1.5.8

df2['price_usd'] = df2['price_brl'] / 3.19
df2.head()



# 1.5.9

df2=df2.drop('price_brl', axis='columns')



df2.dropna(inplace=True)
df2.head()



# 1.5.10


df = pd.concat([df1, df2])
print("df shape:", df.shape)


# Explore

fig = px.scatter_mapbox(
    df,
    lat= df['lat'],
    lon= df['lon'],
    center={"lat": -14.2, "lon": -51.9},  # Map will be centered on Brazil
    width=600,
    height=600,
    hover_data=["price_usd"],  # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()


# 1.5.11
summary_stats = df[['area_m2','price_usd']].describe()
summary_stats.head()


# 1.5.12

plt.hist(df["price_usd"])
plt.xlabel("Price [USD]")
plt.ylabel("Frequency")
plt.title("Distribution of Home Prices")
# Don't change the code below
plt.savefig("images/1-5-12.png", dpi=150)


# In[79]:


with open("images/1-5-12.png", "rb") as file:
    wqet_grader.grade("Project 1 Assessment", "Task 1.5.12", file)



# 1.5.13

plt.boxplot(df["area_m2"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Frequency")
plt.title("Distribution of Home Sizes")
# Don't change the code below
plt.savefig("images/1-5-13.png", dpi=150)


# In[81]:


with open("images/1-5-13.png", "rb") as file:
    wqet_grader.grade("Project 1 Assessment", "Task 1.5.13", file)



# 1.5.14

mean_price_by_region = df.groupby("region")["price_usd"].mean().sort_values()
mean_price_by_region



# 1.5.15

mean_price_by_region.plot(kind="bar", xlabel="Region", ylabel="Mean Price [USD]", title="Mean Home Price by Region")
# Don't change the code below
plt.savefig("images/1-5-15.png", dpi=150)



# 1.5.16

df_south = df[df["region"]=="South"]
df_south.head()



# 1.5.17

homes_by_state = df_south["state"].value_counts()
homes_by_state


# 1.5.18


df_south_large = df_south[df_south["state"]=="Rio Grande do Sul"]
df_south_large.head()

plt.scatter(x=df_south_large["area_m2"], y=df_south_large["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Rio Grande do Sul: Price vs. Area")
# Don't change the code below
plt.savefig("images/1-5-18.png", dpi=150)


# 1.5.19

df_south_Santa = df_south[df_south["state"]=="Santa Catarina"]
df_south_Par = df_south[df_south["state"]=="Paraná"]

south_states_corr = {
    "Rio Grande do Sul": df_south_large["area_m2"].corr(
        df_south_large["price_usd"]
    )
}

south_states_corr["Santa Catarina"] = df_south_Santa["area_m2"].corr(df_south_Santa["price_usd"])
south_states_corr["Paraná"] = df_south_Par["area_m2"].corr(df_south_Par["price_usd"])
south_states_corr
print(south_states_corr)





