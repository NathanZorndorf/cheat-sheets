# Example of a decorator

#Write code here
def wrapper(function):

    def fun(lst):
        
        result = function(lst)
        
        sorted_list = sorted(result)
            
        return sorted_list
        
    return fun

    
@wrapper
def convert_to_phone(num_list):
    output = []

    for phone_num in num_list:
        output.append('+1' + ' ' + str(phone_num[-10:-7]) + ' ' + str(phone_num[-7:-4]) + ' ' + phone_num[-4:])

    return output 


if __name__ == '__main__':

	phone_nums = ['06502505121', '+19658764040']

	convert_to_phone(phone_nums)
