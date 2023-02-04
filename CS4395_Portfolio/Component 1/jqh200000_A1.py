# Jonathan Ho
# CS 4395
'''This python script takes in a .csv file with specific fields.
The data is processed, pickled, and read from the pickle file to print out all the data within the csv.'''

import sys
import os
import re
import pickle

# Person class which would be filled in by the data
class Person:
    # Defines the variables the object would fill with given arguments
    def __init__(self, last, first, mi, id, phone):
        self.last = last
        self.first = first
        self.mi = mi
        self.id = id
        self.phone = phone

    # Prints the contents of the variables
    def display(self):
        print('Employee id:', self.id)
        print('\t', self.first, self.mi, self.last)
        print('\t', self.phone)

# Function to verify the data and correct any fields that are incorrect
def process_file(file):
    # Read in file and create a new dictionary
    f = open(file, 'r')
    next(f)
    person_list = dict()

    # For loop that reads in all data until there is no more data left
    for line in f:
        data_line = line.split(",")

        # Capitalize last and first name in data correctly
        data_line[0] = data_line[0].capitalize()
        data_line[1] = data_line[1].capitalize()

        # Capitalize letter in middle initial or place an X if there is none
        if data_line[2]:
            data_line[2] = data_line[2].upper()
        else:
            data_line[2] = 'X'

        # Check if person ID is valid
        check_ID = data_line[3].upper()
        match_ID = re.match(r'^[A-Z]{2}\d{4}$', check_ID)

        # Prompt for a valid ID until correct format is entered
        while not match_ID:
            print('ID invalid:', data_line[3].upper())
            print('ID is two letters followed by 4 digits')
            check_ID = input('Please enter a valid id: ')
            match_ID = re.match(r'^[A-Z]{2}\d{4}$', check_ID)
        data_line[3] = check_ID.upper()

        # Modify Phone Number to exclude newline character
        check_phone = re.sub(r'\n', '', data_line[4])
        match_phone = re.match(r'^\d{3}\-\d{3}\-\d{4}$', check_phone)

        # Prompt for a valid phone number until correct format is entered
        while not match_phone:
            print('Phone', check_phone, 'is invalid')
            print('Enter phone number in form 123-456-7890')
            check_phone = input('Enter phone number: ')
            match_phone = re.match(r'^\d{3}\-\d{3}\-\d{4}$', check_phone)
        data_line[4] = check_phone

        # Create person to store data
        correct_person = Person(data_line[0], data_line[1], data_line[2], data_line[3], data_line[4])

        # Check if ID is not duplicate
        if data_line[3] in person_list.keys():
            print("Error: ID already exists within list")
        else:
            person_list[data_line[3]] = correct_person

    f.close()
    return person_list

# Function to read in a csv file with the data
def read_csv(filepath):
    with open(os.path.join(os.getcwd(), filepath), 'r') as fp:
        text_in = fp.read()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Read in file
        file_name = sys.argv[1]
        read_csv(file_name)
        print('')

        # Create a Person dictionary of the data
        person_dict = process_file(file_name)

        # Pickle the dictionary
        pickle.dump(person_dict, open('emp_list.p', 'wb'))

        # Read in a dictionary from the pickle made before
        emp_list = pickle.load(open('emp_list.p', 'rb'))

        # Print the list of employees from the dictionary from the pickle file
        print("\n\nEmployee list:\n")
        for key, value in emp_list.items():
            value.display()
            print('')
    else:
        print("Invalid file name")
