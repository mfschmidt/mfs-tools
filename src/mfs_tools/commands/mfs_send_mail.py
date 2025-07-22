#!/usr/bin/env python3

import os.path
import base64
from email.message import EmailMessage
import argparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class App:

    def __init__(self, args):

        # Define the scopes that your application will need.
        # For sending emails,
        # 'https://www.googleapis.com/auth/gmail.send' is required.
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.send', ]
        self.credential_file = os.path.expanduser("~/.ssh/.gmail_credentials.json")
        self.token_file = os.path.expanduser("~/.ssh/.gmail_token.json")

        # Handle arguments
        self.args = args

        # Get the authenticated Gmail service
        self.service = self.get_gmail_service()

    def get_credential_file(self):
        """
        Look in typical places for the credentials.
        """

        for token_path in [
            self.credential_file,
            "./credentials.json",
        ]:
            if os.path.exists(token_path):
                return token_path
        return None

    def get_gmail_service(self):
        """
        Authenticates with the Gmail API and returns a service object.
        """
        creds = None
        # The file token.json stores the user's access and refresh tokens, and
        # is created automatically when the authorization flow completes for
        # the first time.
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(
                self.token_file, self.SCOPES
            )
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.get_credential_file(), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        return build('gmail', 'v1', credentials=creds)


    def create_and_send_email(self):
        """
        Creates and sends an email message.
        """
        try:
            message = EmailMessage()

            # Fill in the email details
            message['To'] = self.args.to
            # message['From'] = self.args.sender  # Automatically uses oauth
            message['Subject'] = self.args.subject
            message.set_content(self.args.body)

            # Encode the message in a URL-safe base64 format
            create_message = {
                'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()
            }

            # Send the email
            send_message = (
                self.service.users().messages().send(
                    userId="me", body=create_message
                ).execute()
            )
            print(f'Sent message to {message["To"]} '
                  f'with Message Id: {send_message["id"]}')

        except HttpError as error:
            print(f'An error occurred: {error}')
            send_message = None

        return send_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "to",
        help="The email address to send the message to.",
    )
    parser.add_argument(
        "subject",
        help="The subject of the email.",
    )
    parser.add_argument(
        "body",
        help="The body of the email.",
    )
    app = App(parser.parse_args())
    app.create_and_send_email()


if __name__ == '__main__':
    main()
