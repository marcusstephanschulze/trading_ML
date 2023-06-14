import psycopg2
import pandas as pd
from psycopg2 import sql
from sqlalchemy import create_engine
import sqlalchemy

class Database_conn:

    def __init__(self):
        self.host = 'localhost'
        self.database = 'Trading_Data'
        self.user = 'postgres'
        self.password = 'Belgaraid1!@'
        
        
        # Establish a connection to the PostgreSQL database
        self.conn = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )
    
    def table_exists(self, table_name):

        cur = self.conn.cursor()
        cur.execute(
            "SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'TICKER_DATA' AND tablename  = %s);",
            (table_name,)
        )
        exists = cur.fetchone()[0]

        return exists
    
    def init_db_L0(self, ticker_list):
        
        print('initialising database bronze layer...')
        
        cur = self.conn.cursor()

        #create table "TICKER_LIST"
        cur.execute('DROP TABLE IF EXISTS "TICKER_DATA"."TICKER_LIST";')
        
        cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS "TICKER_DATA"."TICKER_LIST"
                        (
                            "TICKER" character varying COLLATE pg_catalog."default" NOT NULL
                        )
                        TABLESPACE pg_default;
                        """
                    )
        cur.execute(
                        """
                        ALTER TABLE IF EXISTS "TICKER_DATA"."TICKER_LIST"
                        OWNER to postgres;
                        """
                    )
        #populate "TICKER_LIST"
        
        for ticker in ticker_list:
            cur.execute(
                        """
                        INSERT INTO "TICKER_DATA"."TICKER_LIST" ("TICKER")
                        VALUES (%s);
                        """, [ticker]
                        )
        
            # Create tables for ticker_list: bronze layer (L0)
            
            cur.execute('DROP TABLE IF EXISTS "TICKER_DATA"."L0_{}";'.format(ticker))
            
            cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS "TICKER_DATA"."L0_{}"
                            (
                                id integer NOT NULL,
                                "timestamp" timestamp without time zone NOT NULL,
                                open numeric(10,2) NOT NULL,
                                high numeric(10,2) NOT NULL,
                                low numeric(10,2) NOT NULL,
                                close numeric(10,2) NOT NULL,
                                volume numeric(10,2) NOT NULL,
                                CONSTRAINT "L0_{}_pkey" PRIMARY KEY (id)
                            )

                            TABLESPACE pg_default;
                            """
                        .format(ticker,ticker))
            cur.execute(
                            """
                            ALTER TABLE IF EXISTS "TICKER_DATA"."L0_{}"
                            OWNER to postgres;
                            """
                        .format(ticker))
            cur.execute(
                            """
                            COMMENT ON TABLE "TICKER_DATA"."L0_{}"
                            IS 'BRONZE LAYER, DATA IN FROM CSV FILES';
                            """
                        .format(ticker))
            
            self.conn.commit()
            
        print('initialisation of db bronze layer complete')
         
    def init_db_L1(self, df, ticker):
        
        # can only do one at a time - have to look in main.py
        # dependent on process_data() and the columns of the df
        
        column_names = list(df.columns)
        column_types = column_names[2:-1]
        column_types = ['numeric(10,2)']*len(column_types)
        column_types.insert(0, 'integer')
        column_types.insert(1, 'timestamp without time zone')
        column_types.append('character varying')
        
        cur = self.conn.cursor()

        
        columns = ', '.join([f'{column_names[i]} {column_types[i]}' for i in range(len(column_names))])

        cur.execute(f'DROP TABLE IF EXISTS "TICKER_DATA"."L1_{ticker}";')

        # Use the columns variable in the CREATE TABLE statement
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "TICKER_DATA"."L1_{ticker}"
            ({columns}, CONSTRAINT "L1_{ticker}_pkey" PRIMARY KEY (id))
            TABLESPACE pg_default;
        """)
            
            
        cur.execute(
                        """
                        ALTER TABLE IF EXISTS "TICKER_DATA"."L1_{}"
                        OWNER to postgres;
                        """
                    .format(ticker))
        cur.execute(
                        """
                        COMMENT ON TABLE "TICKER_DATA"."L1_{}"
                        IS 'SILVER LAYER, DATA IN FROM CSV FILES';
                        """
                    .format(ticker))
        
        self.conn.commit()
        
        print('data written to silver layer')
    
    def init_db_L2(self, df, ticker):
        
        # can only do one at a time - have to look in main.py
        # dependent on process_data() and the columns of the df
        
        column_names = list(df.columns)
        column_types = column_names[2:-1]
        column_types = ['numeric(10,2)']*len(column_types)
        column_types.insert(0, 'integer')
        column_types.insert(1, 'timestamp without time zone')
        column_types.append('character varying')
        
        cur = self.conn.cursor()

        
        columns = ', '.join([f'{column_names[i]} {column_types[i]}' for i in range(len(column_names))])

        cur.execute(f'DROP TABLE IF EXISTS "TICKER_DATA"."L2_{ticker}";')

        # Use the columns variable in the CREATE TABLE statement
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "TICKER_DATA"."L2_{ticker}"
            ({columns}, CONSTRAINT "L2_{ticker}_pkey" PRIMARY KEY (id))
            TABLESPACE pg_default;
        """)
            
            
        cur.execute(
                        """
                        ALTER TABLE IF EXISTS "TICKER_DATA"."L2_{}"
                        OWNER to postgres;
                        """
                    .format(ticker))
        cur.execute(
                        """
                        COMMENT ON TABLE "TICKER_DATA"."L2_{}"
                        IS 'SILVER LAYER, DATA IN FROM CSV FILES';
                        """
                    .format(ticker))
        
        self.conn.commit()
        
        print('data written to delta (L2) layer')
    
    def dataframe_to_db(self, df, table_name):
        schema_name = 'TICKER_DATA'
        table_name = f'"{schema_name}"."{table_name}"'
        engine = create_engine('postgresql+psycopg2://', creator=lambda: self.conn)
        with engine.connect() as conn:
            conn.execute(f'SET search_path TO "{schema_name}", public;')
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f'Successfully wrote "{table_name}" to database')
      
    def ticker_list(self):
        
        # Create a cursor object
        cur = self.conn.cursor()

        # Execute a SELECT statement on the TICKER_LIST table
        cur.execute('SELECT * FROM "TICKER_DATA"."TICKER_LIST"')

        # Fetch all the rows from the result set
        rows = cur.fetchall()

        # Create a Pandas dataframe from the rows
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])

        # Return the dataframe
        return df
    
    def write_to_db(self, df, table_name):
        cur = self.conn.cursor()
        columns = list(df.columns)
        
        for i, row in df.iterrows():
            values = [row[col] for col in columns]
            
            #chatGPT-3.5:
            
            insert_query = sql.SQL("""
                INSERT INTO "TICKER_DATA".{} ({}) VALUES ({})
                """).format(sql.Identifier(table_name),
                            sql.SQL(', ').join(map(sql.Identifier, columns)),
                            sql.SQL(', ').join(sql.Placeholder() * len(columns))
                           )
            cur.execute(insert_query, values)
        
        self.conn.commit()
        print(f"Data written to table {table_name} in database {self.database}")

    def overwrite_to_db(self,df, table_name):
        cur = self.conn.cursor()
        # Clear existing data from the table
        cur.execute(f'DELETE FROM "TICKER_DATA"."{table_name}"')
        
        self.write_to_db(df, table_name)

    def sql_to_df(self, table_name):
        
        query = 'SELECT * FROM "TICKER_DATA".{}'.format(table_name)
        # execute query and convert results to a DataFrame
        df = pd.read_sql(query, self.conn)

        # close connection
        
        return df

    def get_recent_timestamp(self, layer, ticker):
        
        from datetime import datetime
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT MAX("timestamp") FROM "TICKER_DATA"."{}_{}"
        """.format(layer, ticker))
        result = cur.fetchone()[0]
        
        return result
    
        # if result:
        #     recent_timestamp = datetime.strptime(result, '%Y-%m-%d %H:%M:%S')
        #     return recent_timestamp
        
        # return None
    
    def append_to_table(self, filtered_df, table_name):
        
        cur = self.conn.cursor()
        columns = filtered_df.columns.tolist()

        # Generate the INSERT INTO SQL statement
        insert_query = sql.SQL("""
            INSERT INTO "TICKER_DATA".{} ({}) VALUES ({})
            """).format(sql.Identifier(table_name),
                        sql.SQL(', ').join(map(sql.Identifier, columns)),
                        sql.SQL(', ').join(sql.Placeholder() * len(columns))
                       )

        # Insert each row of the filtered DataFrame into the table
        for i, row in filtered_df.iterrows():
            values = [row[col] for col in columns]
            cur.execute(insert_query, values)
        
        self.conn.commit()
        cur.close()
        print(f"Data appended to table {table_name} in database {self.database}")

    def close_conn(self):
        self.conn.close()