from __future__ import print_function

import base64
from email.message import EmailMessage

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.insert']


def gmail_send_message():
    """Create and insert a draft email.
       Print the returned draft's message and id.
       Returns: Draft object, including draft id and message meta data.

      Load pre-authorized user credentials from the environment.
      TODO(developer) - See https://developers.google.com/identity
      for guides on implementing OAuth2 for the application.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        
        from ticker_dataset_extraction import daily_update, process_data
        import pandas as pd
        from pathlib import Path

        #run daily_update to get the dataframe

        daily_df = daily_update('30m')
        daily_df = daily_df[['open', 'high','low','close','volume']]
        
        # change dataframe to string, loop through to print in email 

        df_message = {}
        for index, row in daily_df.iterrows():
            df_message[index] = [str(index),':',str("{:.2f}".format(row['close']))]

        print_list = []
        for i in list(df_message.keys()):
            df_message[i] = '     '.join(df_message[i])

        for i in list(df_message.keys()):
            print_list.append(df_message[i] + '     :     recommendation (coming soon)')

        print_list.insert(0, 'symbol: close value: recommendation (coming soon)')
        print_list.insert(1, ' ')
        a = '\n'.join(print_list)
    

        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage()

        prediction_status = 'feature coming soon'
        
        body = a
        message.set_content(body)

        message['To'] = 'miguelmarinhonokia@gmail.com'
        message['From'] = 'marcus.stephan.schulze@gmail.com'
        message['Subject'] = 'Automated email'

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()) \
            .decode()

        create_message = {
            'raw': encoded_message
        }
        # pylint: disable=E1101
        send_message = (service.users().messages().send
                        (userId="me", body=create_message).execute())
        print(F'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        send_message = None
    return send_message


if __name__ == '__main__':
    gmail_send_message()
